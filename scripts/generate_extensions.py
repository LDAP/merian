#!/usr/bin/env python3
"""
Generate C++ extension utility functions from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. get_extension_dependencies() - returns dependencies for a given extension
2. get_extension_property_types() - returns property types for an extension
3. get_api_version_property_types() - returns property types for an API version

This is separate from properties/features to keep concerns separated.
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

    vk_name: str  # e.g., VkPhysicalDevicePushDescriptorPropertiesKHR
    cpp_name: str  # e.g., PhysicalDevicePushDescriptorPropertiesKHR
    stype: str  # e.g., VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR
    extension: Optional[str] = None  # e.g., VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
    core_version: Optional[str] = None  # e.g., VK_API_VERSION_1_4 (when it became core)


def parse_vulkan_spec():
    """Download and parse the Vulkan XML specification."""
    print(f"Downloading Vulkan spec {VULKAN_SPEC_VERSION}...")
    with urlopen(VULKAN_SPEC_URL) as response:
        xml_root = ET.parse(response).getroot()
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


def to_camel_case(name: str) -> str:
    """Convert UPPER_SNAKE_CASE to CamelCase for StructureType enum values."""
    # Simple conversion for structure type enums
    parts = name.split("_")
    result = ""
    for part in parts:
        if part:  # Skip empty parts
            result += part[0].upper() + part[1:].lower()
    return result


def find_property_extension_mapping(xml_root) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Find mapping of extensions and API versions to property structure types.
    Returns (ext_to_stypes, version_to_stypes) where values are sType enum names.
    """
    # Maps extension name macro -> list of structure type enum names (e.g., "ePhysicalDevicePushDescriptorPropertiesKHR")
    ext_to_stypes: dict[str, list[str]] = {}
    # Maps API version macro -> list of structure type enum names
    version_to_stypes: dict[str, list[str]] = {}

    # Build extension name map
    ext_name_map = {}
    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
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

        if ext_name and ext_name_macro:
            ext_name_map[ext_name] = ext_name_macro

    # Find properties from extensions
    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
        ext_supported = ext.get("supported", "")
        if "vulkan" not in ext_supported.split(","):
            continue
        if ext.get("platform") is not None:
            continue

        ext_name_macro = ext_name_map.get(ext_name)
        if not ext_name_macro:
            continue

        for req in ext.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if not type_name:
                    continue

                # Check if this type is a property struct
                for typedef in xml_root.findall("types/type"):
                    if typedef.get("name") == type_name:
                        struct_extends = typedef.get("structextends", "")
                        if "VkPhysicalDeviceProperties2" in struct_extends:
                            # Get the sType
                            for member in typedef.findall("member"):
                                member_name_elem = member.find("name")
                                if member_name_elem is not None and member_name_elem.text == "sType":
                                    stype_value = member.get("values")
                                    if stype_value:
                                        # Convert to enum name: VK_STRUCTURE_TYPE_FOO -> eFoo
                                        enum_name = "e" + to_camel_case(stype_value.replace("VK_STRUCTURE_TYPE_", ""))
                                        ext_to_stypes.setdefault(ext_name_macro, []).append(enum_name)
                                    break
                        break

    # Find properties from API versions (core features)
    for feat in xml_root.findall("feature"):
        api = feat.get("api", "")
        if "vulkan" not in api.split(","):
            continue
        feat_name = feat.get("name", "")
        match = re.match(r"VK_(?:BASE_|COMPUTE_|GRAPHICS_)?VERSION_(\d+)_(\d+)", feat_name)
        if not match:
            continue
        major, minor = match.groups()
        api_version = f"VK_API_VERSION_{major}_{minor}"

        for req in feat.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if not type_name:
                    continue

                # Check if this type is a property struct
                for typedef in xml_root.findall("types/type"):
                    if typedef.get("name") == type_name:
                        struct_extends = typedef.get("structextends", "")
                        if "VkPhysicalDeviceProperties2" in struct_extends or type_name == "VkPhysicalDeviceProperties2":
                            # Get the sType
                            if type_name == "VkPhysicalDeviceProperties2":
                                enum_name = "ePhysicalDeviceProperties2"
                                version_to_stypes.setdefault(api_version, []).append(enum_name)
                            else:
                                for member in typedef.findall("member"):
                                    member_name_elem = member.find("name")
                                    if member_name_elem is not None and member_name_elem.text == "sType":
                                        stype_value = member.get("values")
                                        if stype_value:
                                            enum_name = "e" + to_camel_case(stype_value.replace("VK_STRUCTURE_TYPE_", ""))
                                            version_to_stypes.setdefault(api_version, []).append(enum_name)
                                        break
                        break

    return ext_to_stypes, version_to_stypes


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


def generate_header(extensions: list[Extension]) -> str:
    """Generate the vulkan_extensions.hpp header file content."""
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
        "} // namespace merian",
        "",
    ]
    return "\n".join(lines)


def generate_implementation(
    extensions: list[Extension],
    ext_to_stypes: dict[str, list[str]],
    version_to_stypes: dict[str, list[str]],
) -> str:
    """Generate the vulkan_extensions.cpp implementation file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        '#include "merian/vk/utils/vulkan_extensions.hpp"',
        "",
        "#include <cstring>",
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

    # Generate get_extension_property_types
    lines.extend(
        [
            "std::vector<vk::StructureType> get_extension_property_types(const char* name) {",
            "    std::vector<vk::StructureType> result;",
            "",
        ]
    )

    first = True
    for ext_macro in sorted(ext_to_stypes.keys()):
        stypes = ext_to_stypes[ext_macro]
        condition = "if" if first else "} else if"
        first = False
        lines.append(f"    {condition} (std::strcmp(name, {ext_macro}) == 0) {{")
        for stype_enum in sorted(stypes):
            lines.append(f"        result.push_back(vk::StructureType::{stype_enum});")

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

    # Generate get_api_version_property_types
    lines.extend(
        [
            "std::vector<vk::StructureType> get_api_version_property_types(uint32_t vk_api_version) {",
            "    std::vector<vk::StructureType> result;",
            "",
        ]
    )

    # Sort versions naturally
    def version_sort_key(v: str) -> tuple[int, int]:
        match = re.match(r"VK_API_VERSION_(\d+)_(\d+)", v)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)

    for version in sorted(version_to_stypes.keys(), key=version_sort_key):
        stypes = version_to_stypes[version]
        lines.append(f"    if (vk_api_version >= {version}) {{")
        for stype_enum in sorted(stypes):
            lines.append(f"        result.push_back(vk::StructureType::{stype_enum});")
        lines.append("    }")

    lines.extend(
        [
            "",
            "    return result;",
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

    print("Finding property extension mappings...")
    ext_to_stypes, version_to_stypes = find_property_extension_mapping(xml_root)
    print(f"Extensions with properties: {len(ext_to_stypes)}")
    print(f"API versions with properties: {len(version_to_stypes)}")

    print(f"\nGenerating header file: {include_path / 'vulkan_extensions.hpp'}")
    header_content = generate_header(extensions)
    with open(include_path / "vulkan_extensions.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'vulkan_extensions.cpp'}")
    impl_content = generate_implementation(extensions, ext_to_stypes, version_to_stypes)
    with open(out_path / "vulkan_extensions.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
