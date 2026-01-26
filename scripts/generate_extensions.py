#!/usr/bin/env python3
"""
Generate C++ extension dependency lookup and property wrappers from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. A function to get extension dependencies for a given extension name
2. A function to get property structure types for a given extension
3. A function to get property structure types available at a given API version
4. A function to create a property wrapper by structure type

Similar to generate_features.py, this downloads and parses the official
Vulkan specification to generate type-safe C++ code.
"""

import datetime
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

VULKAN_SPEC_VERSION = "v1.4.338"
VULKAN_SPEC_URL = f"https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/{VULKAN_SPEC_VERSION}/xml/vk.xml"

out_path = Path(__file__).parent.parent / "src" / "merian" / "vk" / "utils"
include_path = Path(__file__).parent.parent / "include" / "merian" / "vk" / "utils"
assert out_path.is_dir(), f"Output path does not exist: {out_path}"
assert include_path.is_dir(), f"Include path does not exist: {include_path}"

# Vendor tags for proper casing (loaded from spec)
tags: list[str] = []


def to_camel_case(name: str) -> str:
    """
    Convert UPPER_SNAKE_CASE to CamelCase, matching vulkan.hpp conventions.

    Rules:
    - After underscore: next char stays uppercase
    - After digit: next char stays uppercase
    - Otherwise: lowercase
    - Vendor tags (KHR, EXT, etc.) stay uppercase at the end
    """
    tag = ""
    if (s := name.split("_")[-1]) in tags:
        tag = s
        name = name[: -1 - len(s)]

    result = ""
    last = None
    for c in name:
        if c == "_":
            pass
        elif last is None:
            result += c
        elif last == "_":
            result += c
        elif last.isdigit():
            result += c
        else:
            result += c.lower()
        last = c

    return result + tag


def vk_name_to_cpp_name(vk_name: str) -> str:
    """Convert VkPhysicalDeviceFoo to PhysicalDeviceFoo."""
    return vk_name.removeprefix("Vk")


def get_stype_from_name(name: str) -> str:
    """
    Convert a struct name to its sType enum value.
    e.g., VkPhysicalDevicePushDescriptorProperties ->
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES
    """
    result = "VK_STRUCTURE_TYPE_"
    prev_lower = False
    prev_digit = False

    for c in name.removeprefix("Vk"):
        if c.isupper() and (prev_lower or prev_digit):
            result += "_"
        elif c.isdigit() and prev_lower:
            result += "_"
        result += c.upper()
        prev_lower = c.islower()
        prev_digit = c.isdigit()

    return result


@dataclass
class ExtensionDep:
    """Represents a dependency that can be either an extension or a Vulkan version."""

    extension: Optional[str] = (
        None  # e.g., VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    )
    version: Optional[str] = None  # e.g., VK_API_VERSION_1_1


@dataclass
class Extension:
    """Represents a Vulkan extension."""

    name: str  # e.g., VK_KHR_swapchain
    name_macro: str  # e.g., VK_KHR_SWAPCHAIN_EXTENSION_NAME
    dependencies: list[list[ExtensionDep]] = field(default_factory=list)  # OR of ANDs


@dataclass
class PropertyStruct:
    """Represents a Vulkan property structure that extends VkPhysicalDeviceProperties2."""

    vk_name: str  # e.g., VkPhysicalDevicePushDescriptorProperties
    cpp_name: str  # e.g., PhysicalDevicePushDescriptorProperties
    stype: str  # e.g., VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES
    extension: Optional[str] = None  # e.g., VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
    core_version: Optional[str] = None  # e.g., VK_API_VERSION_1_4 (when it became core)


def parse_vulkan_spec():
    """Download and parse the Vulkan XML specification."""
    global tags
    print(f"Downloading Vulkan spec {VULKAN_SPEC_VERSION}...")
    with urlopen(VULKAN_SPEC_URL) as response:
        xml_root = ET.parse(response).getroot()

    # Load vendor tags for proper casing
    tags = [i.get("name") for i in xml_root.findall("tags/tag")]  # pyright: ignore[reportAssignmentType]
    return xml_root


def parse_depends(
    depends: str, ext_name_map: dict[str, str]
) -> list[list[ExtensionDep]]:
    """
    Parse the depends attribute into a list of OR'd dependencies,
    where each OR'd item is a list of AND'd dependencies.

    Examples:
    - "VK_KHR_get_physical_device_properties2,VK_VERSION_1_1" -> [[dep1], [ver1_1]]
    - "(VK_KHR_a+VK_KHR_b),VK_VERSION_1_2" -> [[dep_a, dep_b], [ver1_2]]
    """
    if not depends:
        return []

    result = []

    # Split by comma for OR (but not inside parentheses)
    # Simple approach: split by ',' that's not inside parentheses
    or_parts = []
    depth = 0
    current = ""
    for c in depends:
        if c == "(":
            depth += 1
            current += c
        elif c == ")":
            depth -= 1
            current += c
        elif c == "," and depth == 0:
            or_parts.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        or_parts.append(current.strip())

    for or_part in or_parts:
        # Remove outer parentheses if present
        or_part = or_part.strip()
        if or_part.startswith("(") and or_part.endswith(")"):
            or_part = or_part[1:-1]

        # Split by '+' for AND
        and_parts = or_part.split("+")
        and_deps = []

        for and_part in and_parts:
            and_part = and_part.strip()
            # Remove any remaining parentheses
            and_part = and_part.strip("()")

            if and_part.startswith("VK_VERSION_"):
                # It's a version requirement
                match = re.match(r"VK_VERSION_(\d+)_(\d+)", and_part)
                if match:
                    major, minor = match.groups()
                    and_deps.append(
                        ExtensionDep(version=f"VK_API_VERSION_{major}_{minor}")
                    )
            elif and_part.startswith("VK_"):
                # It's an extension requirement
                ext_macro = ext_name_map.get(and_part)
                if ext_macro:
                    and_deps.append(ExtensionDep(extension=ext_macro))

        if and_deps:
            result.append(and_deps)

    return result


def find_extensions(xml_root) -> list[Extension]:
    """Find all Vulkan extensions and their dependencies."""
    extensions = []

    # Build a map of extension name to EXTENSION_NAME macro
    ext_name_map = {}  # VK_KHR_swapchain -> VK_KHR_SWAPCHAIN_EXTENSION_NAME

    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
        ext_supported = ext.get("supported", "")

        # Skip extensions not for Vulkan
        if "vulkan" not in ext_supported.split(","):
            continue

        # Skip platform-specific extensions
        if ext.get("platform") is not None:
            continue

        # Find the EXTENSION_NAME enum
        ext_name_macro = None
        for req in ext.findall("require"):
            for enum_elem in req.findall("enum"):
                enum_name = enum_elem.get("name", "")
                if enum_name.endswith("_EXTENSION_NAME"):
                    ext_name_macro = enum_name
                    break
            if ext_name_macro:
                break

        if ext_name and ext_name_macro:
            ext_name_map[ext_name] = ext_name_macro

    # Now parse extensions with their dependencies
    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
        ext_supported = ext.get("supported", "")

        # Skip extensions not for Vulkan
        if "vulkan" not in ext_supported.split(","):
            continue

        # Skip platform-specific extensions
        if ext.get("platform") is not None:
            continue

        ext_name_macro = ext_name_map.get(ext_name)
        if not ext_name_macro:
            continue

        depends = ext.get("depends", "")
        dependencies = parse_depends(depends, ext_name_map)

        extensions.append(
            Extension(
                name=ext_name,
                name_macro=ext_name_macro,
                dependencies=dependencies,
            )
        )

    return extensions


def find_property_structures(xml_root) -> list[PropertyStruct]:
    """
    Find all structures that extend VkPhysicalDeviceProperties2.
    These are the property structures we generate wrappers for.
    """
    properties = []

    # Build skiplist for platform-specific and non-vulkan types
    skiplist = set()

    for ext in xml_root.findall("extensions/extension"):
        if ext.get("platform") is not None:
            for req in ext.findall("require"):
                for type_elem in req.findall("type"):
                    if name := type_elem.get("name"):
                        skiplist.add(name)

        ext_supported = ext.get("supported", "")
        if "vulkan" not in ext_supported.split(","):
            for req in ext.findall("require"):
                for type_elem in req.findall("type"):
                    if name := type_elem.get("name"):
                        skiplist.add(name)

    # Skip types from non-vulkan API features (e.g., vulkansc)
    for feat in xml_root.findall("feature"):
        if (s := feat.get("api")) is not None and "vulkan" not in s.split(","):
            for req in feat.findall("require"):
                for type_elem in req.findall("type"):
                    if name := type_elem.get("name"):
                        skiplist.add(name)

    # Build map: struct_name -> (ext_name_macro, ext_name)
    ext_type_map = {}
    # Build map: ext_name -> promotedto version
    promotedto_map = {}

    for ext in xml_root.findall("extensions/extension"):
        ext_supported = ext.get("supported", "")
        if "vulkan" not in ext_supported.split(","):
            continue
        if ext.get("platform") is not None:
            continue

        ext_name_macro = None
        for req in ext.findall("require"):
            for enum_elem in req.findall("enum"):
                enum_name = enum_elem.get("name", "")
                if enum_name.endswith("_EXTENSION_NAME"):
                    ext_name_macro = enum_name
                    break
            if ext_name_macro:
                break

        if not ext_name_macro:
            continue

        # Check promotedto attribute
        promotedto = ext.get("promotedto", "")
        if promotedto:
            match = re.match(r"VK_VERSION_(\d+)_(\d+)", promotedto)
            if match:
                major, minor = match.groups()
                promotedto_map[ext.get("name")] = f"VK_API_VERSION_{major}_{minor}"

        for req in ext.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if type_name:
                    ext_type_map[type_name] = (ext_name_macro, ext.get("name"))

    # Build alias map: alias_name -> canonical_name and reverse
    alias_to_canonical = {}
    canonical_to_aliases = {}
    for type_elem in xml_root.findall("types/type"):
        alias = type_elem.get("alias")
        name = type_elem.get("name")
        if alias and name:
            alias_to_canonical[name] = alias
            canonical_to_aliases.setdefault(alias, []).append(name)

    # Propagate ext_type_map through aliases so canonical names also map to extensions
    for alias_name, canonical_name in alias_to_canonical.items():
        if alias_name in ext_type_map and canonical_name not in ext_type_map:
            ext_type_map[canonical_name] = ext_type_map[alias_name]
        elif canonical_name in ext_type_map and alias_name not in ext_type_map:
            ext_type_map[alias_name] = ext_type_map[canonical_name]

    # Build map: struct_name -> VK_API_VERSION from <feature> tags
    feature_type_map = {}
    for feat in xml_root.findall("feature"):
        api = feat.get("api", "")
        if "vulkan" not in api.split(","):
            continue
        feat_name = feat.get("name", "")
        # Match VK_VERSION_X_Y, VK_BASE_VERSION_X_Y, VK_COMPUTE_VERSION_X_Y, VK_GRAPHICS_VERSION_X_Y
        match = re.match(
            r"VK_(?:BASE_|COMPUTE_|GRAPHICS_)?VERSION_(\d+)_(\d+)", feat_name
        )
        if not match:
            continue
        major, minor = match.groups()
        api_version = f"VK_API_VERSION_{major}_{minor}"
        for req in feat.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if type_name:
                    feature_type_map[type_name] = api_version

    # Chain roots to include as property structs themselves
    chain_roots = ["VkPhysicalDeviceProperties2"]

    # Accepted structextends values
    accepted_extends = {"VkPhysicalDeviceProperties2"}

    # Find all structs extending accepted chain roots, plus the chain roots themselves
    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("category") != "struct":
            continue

        vk_name = type_elem.get("name")
        if not vk_name:
            continue

        struct_extends = type_elem.get("structextends", "")
        is_chain_root = vk_name in chain_roots
        extends_accepted = bool(accepted_extends & set(struct_extends.split(",")))

        if not is_chain_root and not extends_accepted:
            continue

        if vk_name in skiplist:
            continue

        # Skip alias definitions
        if type_elem.get("alias") is not None:
            continue

        # Get the sType from the struct members
        stype = None
        for member in type_elem.findall("member"):
            member_name = member.find("name")
            if (
                member_name is not None
                and member_name.text == "sType"
                and member.get("values")
            ):
                stype = member.get("values")
                break

        if not stype:
            stype = get_stype_from_name(vk_name)

        cpp_name = vk_name_to_cpp_name(vk_name)

        # Determine extension and core_version
        extension = None
        core_version = None

        ext_info = ext_type_map.get(vk_name)
        if ext_info:
            ext_name_macro, ext_name = ext_info
            extension = ext_name_macro
            core_version = promotedto_map.get(ext_name)

        # Feature tags override / supplement
        if vk_name in feature_type_map:
            core_version = feature_type_map[vk_name]

        # VkPhysicalDeviceVulkanXXProperties: derive version from name
        # e.g., VkPhysicalDeviceVulkan11Properties -> VK_API_VERSION_1_1
        vulkan_ver_match = re.match(
            r"VkPhysicalDeviceVulkan(\d)(\d)Properties$", vk_name
        )
        if vulkan_ver_match:
            vmajor, vminor = vulkan_ver_match.groups()
            core_version = f"VK_API_VERSION_{vmajor}_{vminor}"

        properties.append(
            PropertyStruct(
                vk_name=vk_name,
                cpp_name=cpp_name,
                stype=stype,
                extension=extension,
                core_version=core_version,
            )
        )

    return properties


def generate_header(
    extensions: list[Extension], properties: list[PropertyStruct]
) -> str:
    """Generate the extensions.hpp header file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        "#pragma once",
        "",
        '#include "vulkan/vulkan.hpp"',
        "",
        "#include <cstdint>",
        "#include <memory>",
        "#include <string>",
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "/**",
        " * @brief Get the list of extension dependencies for a given extension.",
        " * @param name The extension name macro (e.g., VK_KHR_SWAPCHAIN_EXTENSION_NAME).",
        " * @param vk_api_version The Vulkan API version being used.",
        " * @return Vector of extension name strings that are required dependencies.",
        " *         Returns empty if no dependencies or if dependencies are satisfied by the API version.",
        " */",
        "std::vector<const char*> get_extension_dependencies(const char* name, uint32_t vk_api_version);",
        "",
        "/**",
        " * @brief Base class for Vulkan property structures.",
        " *",
        " * Provides a uniform interface for querying physical device properties.",
        " * Wraps a specific VkPhysicalDevice*Properties* structure.",
        " */",
        "class Property {",
        "  public:",
        "    virtual ~Property() = default;",
        "    virtual std::string get_name() const = 0;",
        "    virtual vk::StructureType get_structure_type() const = 0;",
        "    virtual void* get_structure_ptr() = 0;",
        "    virtual void set_pnext(void* p_next) = 0;",
        "};",
        "",
        "using PropertyHandle = std::shared_ptr<Property>;",
        "",
        "/**",
        " * @brief Get the property structure types provided by an extension.",
        " * @param name The extension name macro.",
        " * @return Vector of vk::StructureType values for property structs from this extension.",
        " */",
        "std::vector<vk::StructureType> get_extension_property_types(const char* name);",
        "",
        "/**",
        " * @brief Get all property structure types available at a given API version.",
        " * @param vk_api_version The Vulkan API version.",
        " * @return Vector of vk::StructureType values for core property structs (cumulative).",
        " */",
        "std::vector<vk::StructureType> get_api_version_property_types(uint32_t vk_api_version);",
        "",
        "/**",
        " * @brief Create a Property instance by Vulkan structure type.",
        " * @param stype The vk::StructureType of the property.",
        " * @return A shared pointer to the Property, or nullptr if not found.",
        " */",
        "PropertyHandle get_property(vk::StructureType stype);",
        "",
        "} // namespace merian",
        "",
    ]
    return "\n".join(lines)


def generate_implementation(
    extensions: list[Extension], properties: list[PropertyStruct]
) -> str:
    """Generate the extensions.cpp implementation file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        '#include "merian/vk/utils/extensions.hpp"',
        "",
        "#include <cstring>",
        "#include <typeinfo>",
        "#include <vulkan/vulkan.h>",
        "",
        "namespace merian {",
        "",
        "std::vector<const char*> get_extension_dependencies(const char* name, uint32_t vk_api_version) {",
        "    std::vector<const char*> result;",
        "",
    ]

    # Generate if-else chain for each extension
    first = True
    for ext in sorted(extensions, key=lambda e: e.name):
        if not ext.dependencies:
            continue

        condition = "if" if first else "} else if"
        first = False

        lines.append(f"    {condition} (std::strcmp(name, {ext.name_macro}) == 0) {{")

        # Analyze dependencies:
        # - Find version-only OR groups (can satisfy deps without extensions)
        # - Find extension OR groups (need extensions)
        version_only_groups = []
        extension_groups = []

        for and_group in ext.dependencies:
            has_extension = any(d.extension for d in and_group)
            if has_extension:
                extension_groups.append(and_group)
            else:
                # Version-only group
                versions = [d.version for d in and_group if d.version]
                if versions:
                    version_only_groups.append(versions[0])

        if version_only_groups and extension_groups:
            # We have both version alternatives and extension alternatives
            # If any version is satisfied, no extensions needed
            # Sort versions to check highest first (most likely to be satisfied)
            version_only_groups.sort(reverse=True)
            version_check = " && ".join(
                f"vk_api_version < {v}" for v in version_only_groups
            )
            lines.append(f"        if ({version_check}) {{")
            # Use the first extension group
            for dep in extension_groups[0]:
                if dep.extension:
                    lines.append(f"            result.push_back({dep.extension});")
            lines.append("        }")
        elif version_only_groups:
            # Only version deps - nothing to add (version satisfies it)
            lines.append("        (void)vk_api_version; // Satisfied by Vulkan version")
        elif extension_groups:
            # Only extension deps - add them
            lines.append("        (void)vk_api_version;")
            for dep in extension_groups[0]:
                if dep.extension:
                    lines.append(f"        result.push_back({dep.extension});")

    if not first:
        lines.append("    }")

    lines.extend(
        [
            "",
            "    return result;",
            "}",
            "",
        ]
    )

    # --- PropertyImpl template ---
    lines.extend(
        [
            "// " + "-" * 70,
            "// Property implementation template",
            "// " + "-" * 70,
            "",
            "template <typename T, vk::StructureType SType>",
            "class PropertyImpl : public Property {",
            "  public:",
            "    std::string get_name() const override { return vk::to_string(SType); }",
            "    vk::StructureType get_structure_type() const override { return SType; }",
            "    void* get_structure_ptr() override { return &data; }",
            "    void set_pnext(void* p_next) override { data.pNext = p_next; }",
            "    T& get() { return data; }",
            "    const T& get() const { return data; }",
            "  private:",
            "    T data{};",
            "};",
            "",
        ]
    )

    # --- get_extension_property_types ---
    lines.extend(
        [
            "// " + "-" * 70,
            "// get_extension_property_types",
            "// " + "-" * 70,
            "",
            "std::vector<vk::StructureType> get_extension_property_types(const char* name) {",
            "    std::vector<vk::StructureType> result;",
            "",
        ]
    )

    # Group properties by extension
    ext_to_props: dict[str, list[PropertyStruct]] = {}
    for prop in properties:
        if prop.extension:
            ext_to_props.setdefault(prop.extension, []).append(prop)

    first = True
    for ext_macro in sorted(ext_to_props.keys()):
        props = ext_to_props[ext_macro]
        condition = "if" if first else "} else if"
        first = False
        lines.append(f"    {condition} (std::strcmp(name, {ext_macro}) == 0) {{")
        for prop in sorted(props, key=lambda p: p.cpp_name):
            stype_enum = prop.stype.replace("VK_STRUCTURE_TYPE_", "")
            stype_camel = "e" + to_camel_case(stype_enum)
            lines.append(f"        result.push_back(vk::StructureType::{stype_camel});")

    if not first:
        lines.append("    }")

    lines.extend(
        [
            "",
            "    return result;",
            "}",
            "",
        ]
    )

    # --- get_api_version_property_types ---
    lines.extend(
        [
            "// " + "-" * 70,
            "// get_api_version_property_types",
            "// " + "-" * 70,
            "",
            "std::vector<vk::StructureType> get_api_version_property_types(uint32_t vk_api_version) {",
            "    std::vector<vk::StructureType> result;",
            "",
        ]
    )

    # Group properties by core_version
    version_to_props: dict[str, list[PropertyStruct]] = {}
    for prop in properties:
        if prop.core_version:
            version_to_props.setdefault(prop.core_version, []).append(prop)

    # Sort versions naturally
    def version_sort_key(v: str) -> tuple[int, int]:
        match = re.match(r"VK_API_VERSION_(\d+)_(\d+)", v)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)

    for version in sorted(version_to_props.keys(), key=version_sort_key):
        props = version_to_props[version]
        lines.append(f"    if (vk_api_version >= {version}) {{")
        for prop in sorted(props, key=lambda p: p.cpp_name):
            stype_enum = prop.stype.replace("VK_STRUCTURE_TYPE_", "")
            stype_camel = "e" + to_camel_case(stype_enum)
            lines.append(f"        result.push_back(vk::StructureType::{stype_camel});")
        lines.append("    }")

    lines.extend(
        [
            "",
            "    return result;",
            "}",
            "",
        ]
    )

    # --- get_property ---
    lines.extend(
        [
            "// " + "-" * 70,
            "// get_property",
            "// " + "-" * 70,
            "",
            "PropertyHandle get_property(vk::StructureType stype) {",
            "    switch (stype) {",
        ]
    )

    for prop in sorted(properties, key=lambda p: p.cpp_name):
        stype_enum = prop.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum)
        lines.append(f"        case vk::StructureType::{stype_camel}:")
        lines.append(
            f"            return std::make_shared<PropertyImpl<vk::{prop.cpp_name}, vk::StructureType::{stype_camel}>>();"
        )

    lines.extend(
        [
            "        default:",
            "            return nullptr;",
            "    }",
            "}",
            "",
            "} // namespace merian",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    xml_root = parse_vulkan_spec()

    print("Finding extensions...")
    extensions = find_extensions(xml_root)
    print(f"Found {len(extensions)} extensions")

    # Count extensions with dependencies
    with_deps = [e for e in extensions if e.dependencies]
    print(f"Extensions with dependencies: {len(with_deps)}")

    # Print some examples
    print("\nExample extensions with dependencies:")
    for ext in with_deps[:5]:
        print(f"  - {ext.name}:")
        for or_group in ext.dependencies:
            deps_str = " AND ".join(d.extension or d.version or "?" for d in or_group)
            print(f"      OR: {deps_str}")

    print("\nFinding property structures...")
    properties = find_property_structures(xml_root)
    print(f"Found {len(properties)} property structures")

    with_ext = [p for p in properties if p.extension]
    with_version = [p for p in properties if p.core_version]
    print(f"  With extension: {len(with_ext)}")
    print(f"  With core version: {len(with_version)}")

    print("\nExample property structures:")
    for prop in properties[:5]:
        print(f"  - {prop.cpp_name}: stype={prop.stype}")
        if prop.extension:
            print(f"    Extension: {prop.extension}")
        if prop.core_version:
            print(f"    Core in: {prop.core_version}")

    print(f"\nGenerating header file: {include_path / 'extensions.hpp'}")
    header_content = generate_header(extensions, properties)
    with open(include_path / "extensions.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'extensions.cpp'}")
    impl_content = generate_implementation(extensions, properties)
    with open(out_path / "extensions.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")


if __name__ == "__main__":
    main()
