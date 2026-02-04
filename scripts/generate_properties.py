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
from vulkan_codegen.codegen import (
    build_extension_type_map,
    build_feature_version_map,
    build_struct_aggregation_map,
    generate_file_header,
)
from vulkan_codegen.spec import (
    VULKAN_SPEC_VERSION,
    build_skiplist,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)

out_path, include_path = get_output_paths()


def determine_property_metadata(vk_name, ext_type_map, feature_version_map):
    """Determine extension, core_version, and promotion_version for a property struct."""
    extension = None
    core_version = None
    promotion_version = None

    ext_info = ext_type_map.get(vk_name)
    if ext_info:
        ext_name_macro, ext_name, ext_promotion_version = ext_info
        extension = ext_name_macro
        promotion_version = ext_promotion_version
        core_version = ext_promotion_version

    if vk_name in feature_version_map:
        core_version = feature_version_map[vk_name]

    # VkPhysicalDeviceVulkanXXProperties: derive version from name
    vulkan_ver_match = re.match(r"VkPhysicalDeviceVulkan(\d)(\d)Properties$", vk_name)
    if vulkan_ver_match:
        vmajor, vminor = vulkan_ver_match.groups()
        core_version = f"VK_API_VERSION_{vmajor}_{vminor}"

    return extension, core_version, promotion_version


def find_property_structures(xml_root, tags) -> list[PropertyStruct]:
    """Find all structures that extend VkPhysicalDeviceProperties2."""
    properties = []
    skiplist = build_skiplist(xml_root)

    # Extension map now handles aliases internally during building
    ext_type_map = build_extension_type_map(xml_root)
    feature_version_map = build_feature_version_map(xml_root)

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
        extension, core_version, promotion_version = determine_property_metadata(
            vk_name, ext_type_map, feature_version_map
        )

        properties.append(
            PropertyStruct(
                vk_name=vk_name,
                cpp_name=cpp_name,
                stype=stype,
                extension=extension,
                core_version=core_version,
                promotion_version=promotion_version,
            )
        )

    # Build aggregation map using unified function
    aggregation_map = build_struct_aggregation_map(xml_root, "Properties")

    # Build reverse map
    aggregates_map = {}  # {VkPhysicalDeviceVulkan11Properties: [list of structs]}
    for individual, aggregate in aggregation_map.items():
        if aggregate not in aggregates_map:
            aggregates_map[aggregate] = []
        aggregates_map[aggregate].append(individual)

    # Populate PropertyStruct fields
    for prop in properties:
        if prop.vk_name in aggregation_map:
            prop.aggregated_by = aggregation_map[prop.vk_name]

        if prop.vk_name in aggregates_map:
            prop.aggregates = aggregates_map[prop.vk_name]

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
            "    /// Returns the effective API version of the physical device, that is the minimum of the",
            "    /// targeted version and the supported version.",
            "    uint32_t get_vk_api_version() const {",
            "        return effective_vk_api_version;",
            "    }",
            "",
            "    /// Returns the physical device's supported API version. The effective",
            "    /// version for device use (get_vk_api_version) might be lower.",
            "    uint32_t get_physical_device_vk_api_version() const {",
            "        return m_properties2.properties.apiVersion;",
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
            "",
            "    /// Effective API version (minimum of targeted and physical device version)",
            "    uint32_t effective_vk_api_version = 0;",
            "};",
        ]
    )

    return lines


def generate_header(extensions, properties: list[PropertyStruct]) -> str:
    """Generate the vulkan_properties.hpp header file content."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
        "#pragma once",
        "",
        '#include "vulkan/vulkan.hpp"',
        "",
        "#include <memory>",
        "#include <unordered_set>",
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "// Forward declaration",
        "class Instance;",
        "using InstanceHandle = std::shared_ptr<Instance>;",
        "",
        "std::vector<vk::StructureType> get_api_version_property_types(uint32_t vk_api_version);",
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
    lines = [
        "VulkanProperties::VulkanProperties(const vk::PhysicalDevice& physical_device,",
        "                                   const InstanceHandle& instance) {",
        "    // Calculate effective API version (minimum of targeted and physical device version)",
        "    const uint32_t device_version = physical_device.getProperties().apiVersion;",
        "    effective_vk_api_version = std::min(instance->get_target_vk_api_version(), device_version);",
        "",
        "    // Enumerate supported extensions",
        "    std::unordered_set<std::string> extensions;",
        "    for (const auto& ext : physical_device.enumerateDeviceExtensionProperties()) {",
        "        extensions.insert(ext.extensionName);",
        "    }",
        "",
        "    // Query ALL supported property structs to populate their values",
        "    // Both VkPhysicalDeviceVulkan{XX}Properties and individual promoted structs are queried.",
        "    void* chain_tail = nullptr;",
        "",
    ]

    # Sort all properties alphabetically
    sorted_properties = sorted(properties, key=lambda p: p.cpp_name)

    for prop in sorted_properties:
        member_name = generate_member_name(prop.cpp_name)
        lines.append(f"    // {prop.cpp_name}")

        # Build condition: promotion_version (when promoted to core) OR extension
        # For VulkanXXProperties structs, use core_version since they have no extension
        conditions = []

        # Prefer promotion_version over core_version to avoid duplicates
        if prop.promotion_version:
            # Struct was promoted from an extension - check promotion version OR extension
            conditions.append(f"effective_vk_api_version >= {prop.promotion_version}")
            if prop.extension:
                conditions.append(f"extensions.contains({prop.extension})")
        elif prop.core_version:
            # Core struct (like VulkanXXProperties) - only check core version
            conditions.append(f"effective_vk_api_version >= {prop.core_version}")
        elif prop.extension:
            # Extension-only struct - check extension
            conditions.append(f"extensions.contains({prop.extension})")

        if conditions:
            condition = " || ".join(conditions)
            lines.extend([
                f"    if ({condition}) {{",
                f"        {member_name}.pNext = chain_tail;",
                f"        chain_tail = &{member_name};",
                f"        available_structs.insert({member_name}.sType);",
                "    }",
                "",
            ])
        else:
            lines.extend([
                f"    {member_name}.pNext = chain_tail;",
                f"    chain_tail = &{member_name};",
                f"    available_structs.insert({member_name}.sType);",
                "",
            ])

    lines.extend([
        "    // Query properties from device",
        "    m_properties2.pNext = chain_tail;",
        "    physical_device.getProperties2(&m_properties2);",
        "    available_structs.insert(vk::StructureType::ePhysicalDeviceProperties2);",
        "}",
        "",
    ])

    return lines


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


def generate_api_version_property_types(xml_root, tags) -> list[str]:
    """Generate get_api_version_property_types() implementation."""
    from vulkan_codegen.naming import to_camel_case

    version_to_stypes = {}

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
                if not type_name:
                    continue

                for typedef in xml_root.findall("types/type"):
                    if typedef.get("name") == type_name:
                        struct_extends = typedef.get("structextends", "")
                        if "VkPhysicalDeviceProperties2" in struct_extends:
                            for member in typedef.findall("member"):
                                member_name_elem = member.find("name")
                                if (
                                    member_name_elem is not None
                                    and member_name_elem.text == "sType"
                                ):
                                    stype_value = member.get("values")
                                    if stype_value:
                                        enum_name = "e" + to_camel_case(
                                            stype_value.replace(
                                                "VK_STRUCTURE_TYPE_", ""
                                            ),
                                            tags,
                                        )
                                        version_to_stypes.setdefault(
                                            api_version, []
                                        ).append(enum_name)
                                    break
                        break

    lines = [
        "std::vector<vk::StructureType> get_api_version_property_types(uint32_t vk_api_version) {",
        "    std::vector<vk::StructureType> result;",
        "",
    ]

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

    lines.extend(["", "    return result;", "}", ""])
    return lines


def generate_implementation(extensions, properties: list[PropertyStruct], tags, xml_root) -> str:
    """Generate the vulkan_properties.cpp implementation file content."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
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

    lines.extend(generate_api_version_property_types(xml_root, tags))
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
    impl_content = generate_implementation(extensions, properties, tags, xml_root)
    with open(out_path / "vulkan_properties.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Generated {len(properties)} property struct members")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
