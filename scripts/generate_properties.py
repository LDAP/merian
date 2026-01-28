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

from vulkan_codegen.models import PropertyStruct
from vulkan_codegen.naming import (
    generate_getter_name,
    generate_member_name,
    get_stype_from_name,
    to_camel_case,
    vk_name_to_cpp_name,
)
from vulkan_codegen.parsing import find_extensions
from vulkan_codegen.spec import (
    VULKAN_SPEC_VERSION,
    build_skiplist,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)

out_path, include_path = get_output_paths()


def build_extension_type_map(xml_root):
    """Build map of struct_name -> (ext_name_macro, ext_name)."""
    ext_type_map = {}

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

        for req in ext.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if type_name:
                    ext_type_map[type_name] = (ext_name_macro, ext.get("name"))

    return ext_type_map


def build_promotedto_map(xml_root):
    """Build map of ext_name -> VK_API_VERSION when promoted to core."""
    promotedto_map = {}

    for ext in xml_root.findall("extensions/extension"):
        promotedto = ext.get("promotedto", "")
        if promotedto:
            match = re.match(r"VK_VERSION_(\d+)_(\d+)", promotedto)
            if match:
                major, minor = match.groups()
                promotedto_map[ext.get("name")] = f"VK_API_VERSION_{major}_{minor}"

    return promotedto_map


def build_alias_maps(xml_root):
    """Build bidirectional alias maps for type aliasing."""
    alias_to_canonical = {}
    canonical_to_aliases = {}

    for type_elem in xml_root.findall("types/type"):
        alias = type_elem.get("alias")
        name = type_elem.get("name")
        if alias and name:
            alias_to_canonical[name] = alias
            canonical_to_aliases.setdefault(alias, []).append(name)

    return alias_to_canonical, canonical_to_aliases


def propagate_aliases_to_extension_map(ext_type_map, alias_to_canonical):
    """Propagate extension info through aliases so canonical names also map to extensions."""
    for alias_name, canonical_name in alias_to_canonical.items():
        if alias_name in ext_type_map and canonical_name not in ext_type_map:
            ext_type_map[canonical_name] = ext_type_map[alias_name]
        elif canonical_name in ext_type_map and alias_name not in ext_type_map:
            ext_type_map[alias_name] = ext_type_map[canonical_name]


def build_feature_type_map(xml_root):
    """Build map of struct_name -> VK_API_VERSION from <feature> tags."""
    feature_type_map = {}

    for feat in xml_root.findall("feature"):
        api = feat.get("api", "")
        if "vulkan" not in api.split(","):
            continue
        feat_name = feat.get("name", "")
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

    return feature_type_map


def determine_property_metadata(
    vk_name, ext_type_map, promotedto_map, feature_type_map
):
    """Determine extension and core_version for a property struct."""
    extension = None
    core_version = None

    ext_info = ext_type_map.get(vk_name)
    if ext_info:
        ext_name_macro, ext_name = ext_info
        extension = ext_name_macro
        core_version = promotedto_map.get(ext_name)

    if vk_name in feature_type_map:
        core_version = feature_type_map[vk_name]

    # VkPhysicalDeviceVulkanXXProperties: derive version from name
    vulkan_ver_match = re.match(r"VkPhysicalDeviceVulkan(\d)(\d)Properties$", vk_name)
    if vulkan_ver_match:
        vmajor, vminor = vulkan_ver_match.groups()
        core_version = f"VK_API_VERSION_{vmajor}_{vminor}"

    return extension, core_version


def find_property_structures(xml_root, tags) -> list[PropertyStruct]:
    """Find all structures that extend VkPhysicalDeviceProperties2."""
    properties = []
    skiplist = build_skiplist(xml_root)

    ext_type_map = build_extension_type_map(xml_root)
    promotedto_map = build_promotedto_map(xml_root)
    alias_to_canonical, _ = build_alias_maps(xml_root)
    propagate_aliases_to_extension_map(ext_type_map, alias_to_canonical)
    feature_type_map = build_feature_type_map(xml_root)

    accepted_extends = {"VkPhysicalDeviceProperties2"}

    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("category") != "struct":
            continue

        vk_name = type_elem.get("name")
        if not vk_name:
            continue

        struct_extends = type_elem.get("structextends", "")
        if not (accepted_extends & set(struct_extends.split(","))):
            continue

        if vk_name in skiplist or type_elem.get("alias") is not None:
            continue

        # Get sType from struct members
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
        extension, core_version = determine_property_metadata(
            vk_name, ext_type_map, promotedto_map, feature_type_map
        )

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


def generate_concept_definition(properties: list[PropertyStruct]) -> list[str]:
    """Generate the VulkanPropertyStruct concept definition."""
    lines = [
        "/**",
        " * @brief C++20 concept for validating Vulkan property structure types.",
        " * @tparam T The type to validate.",
        " */",
        "template <typename T>",
        "concept VulkanPropertyStruct = (",
    ]

    type_checks = ["    std::same_as<T, vk::PhysicalDeviceProperties2>"]
    for prop in sorted(properties, key=lambda p: p.cpp_name):
        type_checks.append(f"    std::same_as<T, vk::{prop.cpp_name}>")

    lines.append(" ||\n".join(type_checks))
    lines.append(");")
    lines.append("")

    return lines


def generate_class_declaration(properties: list[PropertyStruct]) -> list[str]:
    """Generate the VulkanProperties class declaration."""
    lines = [
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
        "    // Named getters for each property struct",
        "    const vk::PhysicalDeviceProperties& get_properties() const;",
        "    operator const vk::PhysicalDeviceProperties&() const;",
        "    const vk::PhysicalDeviceProperties2& get_properties2() const;",
        "    operator const vk::PhysicalDeviceProperties2&() const;",
    ]

    for prop in sorted(properties, key=lambda p: p.cpp_name):
        getter_name = generate_getter_name(prop.cpp_name)
        lines.append(f"    const vk::{prop.cpp_name}& {getter_name}() const;")
        lines.append(f"    operator const vk::{prop.cpp_name}&() const;")

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
            "    // Property struct members (~120 structs)",
        ]
    )

    for prop in sorted(properties, key=lambda p: p.cpp_name):
        member_name = generate_member_name(prop.cpp_name)
        lines.append(f"    vk::{prop.cpp_name} {member_name}{{}};")

    lines.extend(
        [
            "",
            "    vk::PhysicalDeviceProperties2 m_properties2{};",
            "",
            "    /// Track which structs are available via extensions or API version",
            "    std::unordered_set<vk::StructureType> available_structs;",
            "};",
        ]
    )

    return lines


def generate_header(extensions, properties: list[PropertyStruct]) -> str:
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

    lines.extend(generate_concept_definition(properties))
    lines.extend(generate_class_declaration(properties))
    lines.extend(
        [
            "",
            "} // namespace merian",
            "",
        ]
    )

    return "\n".join(lines)


def generate_constructor(properties: list[PropertyStruct]) -> list[str]:
    """Generate VulkanProperties constructor implementation."""
    return [
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


def generate_template_get(properties: list[PropertyStruct]) -> list[str]:
    """Generate template get<T>() method and explicit instantiations."""
    lines = [
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
        "// Explicit template instantiations",
        "template const vk::PhysicalDeviceProperties2& VulkanProperties::get<vk::PhysicalDeviceProperties2>() const;",
    ]

    for prop in sorted(properties, key=lambda p: p.cpp_name):
        lines.append(
            f"template const vk::{prop.cpp_name}& VulkanProperties::get<vk::{prop.cpp_name}>() const;"
        )
    lines.append("")

    return lines


def generate_named_getters(properties: list[PropertyStruct], tags) -> list[str]:
    """Generate named getter implementations."""
    lines = [
        "// Named getter implementations",
        "const vk::PhysicalDeviceProperties& VulkanProperties::get_properties() const {",
        "    return m_properties2.properties;",
        "}",
        "VulkanProperties::operator const vk::PhysicalDeviceProperties&() const {",
        "    return get_properties();",
        "}",
        "const vk::PhysicalDeviceProperties2& VulkanProperties::get_properties2() const {",
        "    return m_properties2;",
        "}",
        "VulkanProperties::operator const vk::PhysicalDeviceProperties2&() const {",
        "    return get_properties2();",
        "}",
    ]

    for prop in sorted(properties, key=lambda p: p.cpp_name):
        member_name = generate_member_name(prop.cpp_name)
        getter_name = generate_getter_name(prop.cpp_name)
        stype_enum = prop.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)

        lines.extend(
            [
                f"const vk::{prop.cpp_name}& VulkanProperties::{getter_name}() const {{",
                f"    if (!is_available(vk::StructureType::{stype_camel})) {{",
                "        throw std::runtime_error(fmt::format(",
                f'            "Property struct {prop.cpp_name} not available (extension not supported)"));',
                "    }",
                f"    return {member_name};",
                "}",
                f"VulkanProperties::operator const vk::{prop.cpp_name}&() const {{",
                f"    return {getter_name}();",
                "}",
            ]
        )

    lines.append("")
    return lines


def generate_get_struct_ptr(properties: list[PropertyStruct], tags) -> list[str]:
    """Generate get_struct_ptr switch statement."""
    lines = [
        "const void* VulkanProperties::get_struct_ptr(vk::StructureType stype) const {",
        "    switch (stype) {",
        "        case vk::StructureType::ePhysicalDeviceProperties2:",
        "            return &m_properties2;",
    ]

    for prop in sorted(properties, key=lambda p: p.cpp_name):
        member_name = generate_member_name(prop.cpp_name)
        stype_enum = prop.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)
        lines.extend(
            [
                f"        case vk::StructureType::{stype_camel}:",
                f"            return &{member_name};",
            ]
        )

    lines.extend(
        [
            "        default:",
            "            return nullptr;",
            "    }",
            "}",
            "",
        ]
    )

    return lines


def generate_is_available() -> list[str]:
    """Generate is_available implementation."""
    return [
        "bool VulkanProperties::is_available(vk::StructureType stype) const {",
        "    // Properties2 is always available",
        "    if (stype == vk::StructureType::ePhysicalDeviceProperties2) {",
        "        return true;",
        "    }",
        "    return available_structs.contains(stype);",
        "}",
        "",
    ]


def generate_implementation(extensions, properties: list[PropertyStruct], tags) -> str:
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

    lines.extend(generate_constructor(properties))
    lines.extend(generate_template_get(properties))
    lines.extend(generate_named_getters(properties, tags))
    lines.extend(generate_get_struct_ptr(properties, tags))
    lines.extend(generate_is_available())

    lines.extend(
        [
            "} // namespace merian",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    xml_root = load_vulkan_spec()
    tags = load_vendor_tags(xml_root)

    print("Finding extensions...")
    extensions = find_extensions(xml_root)
    print(f"Found {len(extensions)} extensions")

    with_deps = [e for e in extensions if e.dependencies]
    print(f"Extensions with dependencies: {len(with_deps)}")

    print("\nFinding property structures...")
    properties = find_property_structures(xml_root, tags)
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
    impl_content = generate_implementation(extensions, properties, tags)
    with open(out_path / "vulkan_properties.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Generated {len(properties)} property struct members")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
