#!/usr/bin/env python3
"""
Generate C++ VulkanProperties aggregate class from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. A VulkanProperties class containing all ~120 property structs as members
2. Type-safe template access methods
3. Named getters for each property struct
4. Constructor that queries properties from physical device

Replaces the old PropertyImpl template with a more efficient aggregate design
that eliminates unsafe casting and provides better type safety.

Note: Extension dependency and property lookup functions are in generate_extensions.py
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


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case for member names."""
    result = ""
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            # Insert underscore before uppercase if previous char was lowercase or digit
            if name[i - 1].islower() or name[i - 1].isdigit():
                result += "_"
            # Insert underscore before uppercase if next char is lowercase (handles acronyms)
            elif i + 1 < len(name) and name[i + 1].islower():
                result += "_"
        result += c.lower()
    return result


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
        extends_accepted = bool(accepted_extends & set(struct_extends.split(",")))

        if not extends_accepted:
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


def generate_member_name(cpp_name: str) -> str:
    """Generate member variable name from struct name.

    PhysicalDevicePushDescriptorPropertiesKHR -> m_push_descriptor_properties_khr
    PhysicalDeviceProperties2 -> m_properties2
    """
    # Remove PhysicalDevice prefix
    name = cpp_name.removeprefix("PhysicalDevice")
    # Convert to snake_case
    snake = to_snake_case(name)
    return f"m_{snake}"


def generate_getter_name(cpp_name: str) -> str:
    """Generate getter method name from struct name.

    PhysicalDevicePushDescriptorPropertiesKHR -> get_push_descriptor_properties_khr
    PhysicalDeviceProperties2 -> get_properties_2
    """
    # Remove PhysicalDevice prefix
    name = cpp_name.removeprefix("PhysicalDevice")
    # Convert to snake_case
    snake = to_snake_case(name)
    return f"get_{snake}"


def generate_header(
    extensions: list[Extension], properties: list[PropertyStruct]
) -> str:
    """Generate the vulkan_properties.hpp header file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        "#pragma once",
        "",
        '#include "vulkan/vulkan.hpp"',
        "",
        "#include <memory>",
        "#include <unordered_set>",
        "",
        "namespace merian {",
        "",
        "// Forward declaration",
        "class Instance;",
        "using InstanceHandle = std::shared_ptr<Instance>;",
        "",
    ]

    # Generate VulkanPropertyStruct concept
    lines.append("/**")
    lines.append(
        " * @brief C++20 concept for validating Vulkan property structure types."
    )
    lines.append(" * @tparam T The type to validate.")
    lines.append(" */")
    lines.append("template <typename T>")
    lines.append("concept VulkanPropertyStruct = (")

    # Generate type checks for all properties
    type_checks = []
    type_checks.append("    std::same_as<T, vk::PhysicalDeviceProperties2>")
    for prop in sorted(properties, key=lambda p: p.cpp_name):
        type_checks.append(f"    std::same_as<T, vk::{prop.cpp_name}>")

    lines.append(" ||\n".join(type_checks))
    lines.append(");")
    lines.append("")

    # Generate VulkanProperties class
    lines.extend(
        [
            "/**",
            " * @brief Aggregate class containing all Vulkan property structures.",
            " *",
            " * Provides type-safe access to all ~120 Vulkan property structs without",
            " * unsafe casting or map lookups. Each property struct is stored as a direct member.",
            " *",
            " * Properties can be accessed via:",
            " * - Template method: get<vk::PhysicalDevicePushDescriptorPropertiesKHR>()",
            " * - Named getter: get_push_descriptor_properties_khr()",
            " */",
            "class VulkanProperties {",
            "  public:",
            "    /// Default constructor (zero-initialized properties)",
            "    VulkanProperties() = delete;",
            "",
            "    /// Constructor: query properties from physical device",
            "    VulkanProperties(const vk::PhysicalDevice& physical_device,",
            "                     const InstanceHandle& instance);",
            "",
            "    /// Copy and move",
            "    VulkanProperties(const VulkanProperties&) = default;",
            "    VulkanProperties& operator=(const VulkanProperties&) = default;",
            "    VulkanProperties(VulkanProperties&&) = default;",
            "    VulkanProperties& operator=(VulkanProperties&&) = default;",
            "",
            "    /// Type-safe template access (throws if extension not available)",
            "    template<typename T> requires VulkanPropertyStruct<T>",
            "    const T& get() const;",
            "",
        ]
    )

    # Generate named getters
    lines.append("    // Named getters for each property struct")
    lines.append("    const vk::PhysicalDeviceProperties& get_properties() const;")
    lines.append("    const vk::PhysicalDeviceProperties2& get_properties2() const;")
    for prop in sorted(properties, key=lambda p: p.cpp_name):
        getter_name = generate_getter_name(prop.cpp_name)
        lines.append(f"    const vk::{prop.cpp_name}& {getter_name}() const;")

    lines.extend(
        [
            "",
            "    /// Check if property struct is available (only valid after physical_device constructor)",
            "    bool is_available(vk::StructureType stype) const;",
            "",
            "    template<typename T> requires VulkanPropertyStruct<T>",
            "    bool is_available() const {",
            "        return is_available(T::structureType);",
            "    }",
            "",
            "  private:",
            "    /// Internal: get pointer to struct by StructureType",
            "    const void* get_struct_ptr(vk::StructureType stype) const;",
            "",
        ]
    )

    # Generate member variables
    lines.append("    // Property struct members (~120 structs)")
    for prop in sorted(properties, key=lambda p: p.cpp_name):
        member_name = generate_member_name(prop.cpp_name)
        lines.append(f"    vk::{prop.cpp_name} {member_name}{{}};")
    lines.append("")
    lines.append("    vk::PhysicalDeviceProperties2 m_properties2{};")

    lines.extend(
        [
            "",
            "    /// Track which structs are available via extensions or API version",
            "    std::unordered_set<vk::StructureType> available_structs;",
            "};",
            "",
            "} // namespace merian",
            "",
        ]
    )

    return "\n".join(lines)


def generate_implementation(
    extensions: list[Extension], properties: list[PropertyStruct]
) -> str:
    """Generate the vulkan_properties.cpp implementation file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        '#include "merian/vk/utils/vulkan_properties.hpp"',
        '#include "merian/vk/utils/vulkan_extensions.hpp"',
        '#include "merian/vk/instance.hpp"',
        "",
        "#include <fmt/format.h>",
        "#include <stdexcept>",
        "#include <vulkan/vulkan.h>",
        "",
        "namespace merian {",
        "",
    ]

    # Generate get_extension_dependencies
    # Generate VulkanProperties constructor
    lines.extend(
        [
            "VulkanProperties::VulkanProperties(const vk::PhysicalDevice& physical_device,",
            "                                   const InstanceHandle& instance) {",
            "    // Enumerate supported extensions",
            "    std::unordered_set<std::string> extensions;",
            "    for (const auto& ext : physical_device.enumerateDeviceExtensionProperties()) {",
            "        extensions.insert(ext.extensionName);",
            "    }",
            "",
            "    // Build pNext chain using BaseOutStructure for safe chaining",
            "    vk::BaseOutStructure* chain_tail = nullptr;",
            "",
            "    // Add properties from API version",
            "    for (const auto& stype : get_api_version_property_types(instance->get_vk_api_version())) {",
            "        void* struct_ptr = const_cast<void*>(get_struct_ptr(stype));",
            "        if (struct_ptr != nullptr) {",
            "            auto* base = reinterpret_cast<vk::BaseOutStructure*>(struct_ptr);",
            "            base->pNext = chain_tail;",
            "            chain_tail = base;",
            "            available_structs.insert(stype);",
            "        }",
            "    }",
            "",
            "    // Add properties from extensions",
            "    for (const auto& ext_name : extensions) {",
            "        const auto* const ext_macro = ext_name.c_str();",
            "        for (const auto& stype : get_extension_property_types(ext_macro)) {",
            "            if (available_structs.contains(stype)) continue;  // Already added via API version",
            "            void* struct_ptr = const_cast<void*>(get_struct_ptr(stype));",
            "            if (struct_ptr != nullptr) {",
            "                auto* base = reinterpret_cast<vk::BaseOutStructure*>(struct_ptr);",
            "                base->pNext = chain_tail;",
            "                chain_tail = base;",
            "                available_structs.insert(stype);",
            "            }",
            "        }",
            "    }",
            "",
            "    // Query properties from device",
            "    m_properties2.pNext = chain_tail;",
            "    physical_device.getProperties2(&m_properties2);",
            "}",
            "",
        ]
    )

    # Generate template get<T>() method
    lines.extend(
        [
            "// Template get<T>() implementation",
            "template<typename T> requires VulkanPropertyStruct<T>",
            "const T& VulkanProperties::get() const {",
            "    const auto stype = T::structureType;",
            "    if (!is_available(stype)) {",
            "        throw std::runtime_error(fmt::format(",
            '            "Property struct {} not available (extension not supported)", ',
            "            vk::to_string(stype)));",
            "    }",
            "    const void* ptr = get_struct_ptr(stype);",
            "    if (!ptr) {",
            "        throw std::runtime_error(fmt::format(",
            '            "Unknown property struct type: {}", vk::to_string(stype)));',
            "    }",
            "    return *reinterpret_cast<const T*>(ptr);",
            "}",
            "",
        ]
    )

    # Generate explicit template instantiations
    lines.append("// Explicit template instantiations")
    lines.append(
        "template const vk::PhysicalDeviceProperties2& VulkanProperties::get<vk::PhysicalDeviceProperties2>() const;"
    )
    for prop in sorted(properties, key=lambda p: p.cpp_name):
        lines.append(
            f"template const vk::{prop.cpp_name}& VulkanProperties::get<vk::{prop.cpp_name}>() const;"
        )
    lines.append("")

    # Generate named getters
    lines.append("// Named getter implementations")
    lines.append(
        "const vk::PhysicalDeviceProperties& VulkanProperties::get_properties() const {"
    )
    lines.append("    return m_properties2.properties;")
    lines.append("}")
    lines.append("// Named getter implementations")
    lines.append(
        "const vk::PhysicalDeviceProperties2& VulkanProperties::get_properties2() const {"
    )
    lines.append("    return m_properties2;")
    lines.append("}")

    for prop in sorted(properties, key=lambda p: p.cpp_name):
        member_name = generate_member_name(prop.cpp_name)
        getter_name = generate_getter_name(prop.cpp_name)
        stype_enum = prop.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum)

        lines.append(
            f"const vk::{prop.cpp_name}& VulkanProperties::{getter_name}() const {{"
        )
        lines.append(f"    if (!is_available(vk::StructureType::{stype_camel})) {{")
        lines.append("        throw std::runtime_error(fmt::format(")
        lines.append(
            f'            "Property struct {prop.cpp_name} not available (extension not supported)"));'
        )
        lines.append("    }")
        lines.append(f"    return {member_name};")
        lines.append("}")

    lines.append("")

    # Generate get_struct_ptr
    lines.extend(
        [
            "const void* VulkanProperties::get_struct_ptr(vk::StructureType stype) const {",
            "    switch (stype) {",
        ]
    )

    # Add PhysicalDeviceProperties2
    # Note: PhysicalDeviceProperties doesn't have its own StructureType enum value
    lines.append("        case vk::StructureType::ePhysicalDeviceProperties2:")
    lines.append("            return &m_properties2;")

    for prop in sorted(properties, key=lambda p: p.cpp_name):
        member_name = generate_member_name(prop.cpp_name)
        stype_enum = prop.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum)
        lines.append(f"        case vk::StructureType::{stype_camel}:")
        lines.append(f"            return &{member_name};")

    lines.extend(
        [
            "        default:",
            "            return nullptr;",
            "    }",
            "}",
            "",
        ]
    )

    # Generate is_available
    lines.extend(
        [
            "bool VulkanProperties::is_available(vk::StructureType stype) const {",
            "    // Properties2 is always available",
            "    if (stype == vk::StructureType::ePhysicalDeviceProperties2) {",
            "        return true;",
            "    }",
            "    return available_structs.contains(stype);",
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

    print("\nFinding property structures...")
    properties = find_property_structures(xml_root)
    print(f"Found {len(properties)} property structures")

    with_ext = [p for p in properties if p.extension]
    with_version = [p for p in properties if p.core_version]
    print(f"  With extension: {len(with_ext)}")
    print(f"  With core version: {len(with_version)}")

    print(f"\nGenerating header file: {include_path / 'vulkan_properties.hpp'}")
    header_content = generate_header(extensions, properties)
    with open(include_path / "vulkan_properties.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'vulkan_properties.cpp'}")
    impl_content = generate_implementation(extensions, properties)
    with open(out_path / "vulkan_properties.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Generated {len(properties)} property struct members")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
