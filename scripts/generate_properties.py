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
from vulkan_codegen.parsing import find_extensions, build_extension_map, find_all_structures
from vulkan_codegen.codegen import (
    build_extension_type_map,
    build_feature_version_map,
    generate_file_header,
    generate_stype_switch,
    get_extension,
)
from vulkan_codegen.spec import (
    PROPERTY_STRUCT_BASE,
    VULKAN_SPEC_VERSION,
    build_skiplist,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)

out_path, include_path = get_output_paths()

# Global extension map for accessing extension properties
_extension_map = {}


def find_property_structures(xml_root, tags) -> list[PropertyStruct]:
    """Find all structures that extend VkPhysicalDeviceProperties2."""
    skiplist = build_skiplist(xml_root)

    # Build full extension map
    global _extension_map
    _extension_map = build_extension_map(xml_root, tags)

    # Extract all structs using unified function
    all_structs = find_all_structures(xml_root, tags, skiplist)

    # Filter for property structs (includes base struct and all extensions)
    property_structs = [
        s for s in all_structs
        if s.vk_name == PROPERTY_STRUCT_BASE  # Include VkPhysicalDeviceProperties2 itself
        or PROPERTY_STRUCT_BASE in s.structextends  # Or structs that extend it
    ]

    # Convert VulkanStruct -> PropertyStruct
    # Get core_version from version map for core structs
    from vulkan_codegen.codegen import build_feature_version_map
    version_map = build_feature_version_map(xml_root)

    properties = []
    for vulkan_struct in property_structs:
        # Determine if this is a core struct (introduced in a Vulkan version)
        core_version = version_map.get(vulkan_struct.vk_name)

        properties.append(PropertyStruct(
            vk_name=vulkan_struct.vk_name,
            cpp_name=vulkan_struct.cpp_name,
            stype=vulkan_struct.stype,
            extension_name=vulkan_struct.extension_name,
            structextends=vulkan_struct.structextends,
            aliases=vulkan_struct.aliases,
            is_alias=vulkan_struct.is_alias,
            members=vulkan_struct.members,  # Keep raw members from base class
            core_version=core_version,  # PropertyStruct uses 'core_version' instead of 'required_version'
        ))

    return properties


def _canonical_properties_only(properties: list[PropertyStruct]) -> list[PropertyStruct]:
    """Filter out aliases - only return canonical structs for member/getter generation."""
    return [p for p in properties if not p.is_alias]


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

    # Base struct handled separately + canonical structs only (no aliases)
    type_checks = ["    std::same_as<T, vk::PhysicalDeviceProperties2>"]
    canonical = _canonical_properties_only(properties)
    for prop in sorted(canonical, key=lambda p: p.cpp_name):
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
    ]

    # Only canonical structs get main getters (no aliases)
    canonical = _canonical_properties_only(properties)
    for prop in sorted(canonical, key=lambda p: p.cpp_name):
        getter_name = generate_getter_name(prop.cpp_name)
        lines.append(f"    const vk::{prop.cpp_name}& {getter_name}() const;")
        lines.append(f"    operator const vk::{prop.cpp_name}&() const;")

    lines.extend(
        [
            "",
            "    // Alias getters (for backwards compatibility)",
        ]
    )

    # Generate alias getters from canonical struct's aliases list
    for prop in sorted(canonical, key=lambda p: p.cpp_name):
        for alias_vk_name in prop.aliases:
            alias_cpp_name = vk_name_to_cpp_name(alias_vk_name)
            alias_getter_name = generate_getter_name(alias_cpp_name)
            lines.append(
                f"    const vk::{prop.cpp_name}& {alias_getter_name}() const;  // Alias for {prop.cpp_name}"
            )

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

    # Only canonical structs get member declarations (no aliases)
    canonical = _canonical_properties_only(properties)
    for prop in sorted(canonical, key=lambda p: p.cpp_name):
        member_name = generate_member_name(prop.cpp_name)
        lines.append(f"    vk::{prop.cpp_name} {member_name}{{}};")

    lines.extend(
        [
            "",
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

    # Only canonical properties (no aliases) that extend the base struct
    canonical = _canonical_properties_only(properties)
    extending_properties = [p for p in canonical if PROPERTY_STRUCT_BASE in p.structextends]
    sorted_properties = sorted(extending_properties, key=lambda p: p.cpp_name)

    for prop in sorted_properties:
        member_name = generate_member_name(prop.cpp_name)
        lines.append(f"    // {prop.cpp_name}")

        # Build condition: promotion_version (when promoted to core) OR extension
        # For VulkanXXProperties structs, use core_version since they have no extension
        conditions = []

        ext = get_extension(prop, _extension_map)

        # Prefer promotion_version over core_version to avoid duplicates
        if ext and ext.promotedto:
            # Struct was promoted from an extension - check promotion version OR extension
            conditions.append(f"effective_vk_api_version >= {ext.promotedto}")
            conditions.append(f"extensions.contains({ext.name_macro})")
        elif prop.core_version:
            # Core struct (like VulkanXXProperties) - only check core version
            conditions.append(f"effective_vk_api_version >= {prop.core_version}")
        elif ext:
            # Extension-only struct - check extension
            conditions.append(f"extensions.contains({ext.name_macro})")

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

    # PhysicalDeviceProperties2 is the base - query it last
    properties2 = next((p for p in canonical if p.vk_name == PROPERTY_STRUCT_BASE), None)
    if properties2:
        member_name = generate_member_name(properties2.cpp_name)
        lines.extend([
            f"    // Query properties from device",
            f"    {member_name}.pNext = chain_tail;",
            f"    physical_device.getProperties2(&{member_name});",
            f"    available_structs.insert({member_name}.sType);",
            "}",
            "",
        ])
    else:
        lines.extend(["}",""])

    return lines


def generate_template_get(properties: list[PropertyStruct]) -> list[str]:
    """Generate template get<T>() method and explicit instantiations."""
    lines = [
        "// Template get<T>() implementation",
        "template<typename T> requires VulkanPropertyStruct<T>",
        "const T& VulkanProperties::get() const {",
        "    assert(is_available(T::structureType));",
        "    const void* ptr = get_struct_ptr(T::structureType);",
        "    assert(ptr);",
        "    return *reinterpret_cast<const T*>(ptr);",
        "}",
        "",
        "// Explicit template instantiations",
    ]

    # Only canonical structs get template instantiations (no aliases)
    canonical = _canonical_properties_only(properties)
    for prop in sorted(canonical, key=lambda p: p.cpp_name):
        lines.append(
            f"template const vk::{prop.cpp_name}& VulkanProperties::get<vk::{prop.cpp_name}>() const;"
        )
    lines.append("")

    return lines


def generate_named_getters(properties: list[PropertyStruct], tags) -> list[str]:
    """Generate named getter implementations."""
    lines = [
        "// Named getter implementations",
        "// Special getter for inner properties field",
        "const vk::PhysicalDeviceProperties& VulkanProperties::get_properties() const {",
        "    return m_properties2.properties;",
        "}",
        "VulkanProperties::operator const vk::PhysicalDeviceProperties&() const {",
        "    return get_properties();",
        "}",
    ]

    # Only canonical structs get main getters (no aliases)
    canonical = _canonical_properties_only(properties)
    for prop in sorted(canonical, key=lambda p: p.cpp_name):
        member_name = generate_member_name(prop.cpp_name)
        getter_name = generate_getter_name(prop.cpp_name)
        stype_enum = prop.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)

        lines.extend(
            [
                f"const vk::{prop.cpp_name}& VulkanProperties::{getter_name}() const {{",
                f"    assert(is_available(vk::StructureType::{stype_camel}));",
                f"    return {member_name};",
                "}",
                f"VulkanProperties::operator const vk::{prop.cpp_name}&() const {{",
                f"    return {getter_name}();",
                "}",
            ]
        )

    lines.append("")
    return lines


def generate_alias_getters(properties: list[PropertyStruct]) -> list[str]:
    """Generate alias getter implementations."""
    lines = ["// Alias getter implementations (for backwards compatibility)"]

    # Only canonical structs have aliases to generate getters for
    canonical = _canonical_properties_only(properties)
    for prop in sorted(canonical, key=lambda p: p.cpp_name):
        canonical_getter_name = generate_getter_name(prop.cpp_name)
        for alias_vk_name in prop.aliases:
            alias_cpp_name = vk_name_to_cpp_name(alias_vk_name)
            alias_getter_name = generate_getter_name(alias_cpp_name)
            lines.extend(
                [
                    f"const vk::{prop.cpp_name}& VulkanProperties::{alias_getter_name}() const {{",
                    f"    return {canonical_getter_name}();  // Alias for {prop.cpp_name}",
                    "}",
                ]
            )

    lines.append("")
    return lines


def generate_get_struct_ptr(properties: list[PropertyStruct], tags) -> list[str]:
    """Generate get_struct_ptr switch statement."""
    # Only canonical properties (no aliases) - aliases share sType with canonical
    canonical = _canonical_properties_only(properties)
    return generate_stype_switch(canonical, "VulkanProperties", tags)


def generate_is_available() -> list[str]:
    """Generate is_available implementation."""
    return [
        "bool VulkanProperties::is_available(vk::StructureType stype) const {",
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
                        if PROPERTY_STRUCT_BASE in struct_extends:
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
        "#include <cassert>",
        "#include <vulkan/vulkan.h>",
        "",
        "namespace merian {",
        "",
    ]

    lines.extend(generate_api_version_property_types(xml_root, tags))
    lines.extend(generate_constructor(properties))
    lines.extend(generate_template_get(properties))
    lines.extend(generate_named_getters(properties, tags))
    lines.extend(generate_alias_getters(properties))
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

    with_ext = [p for p in properties if p.extension_name]
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
