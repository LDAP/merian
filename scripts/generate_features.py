#!/usr/bin/env python3
"""
Generate C++ VulkanFeatures aggregate class from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. A VulkanFeatures class containing all ~240 feature structs as members
2. Type-safe template access methods
3. Named getters for each feature struct
4. String-based runtime access methods
5. Constructor that queries features from physical device

Replaces the old polymorphic Feature class hierarchy with a more efficient
aggregate design that eliminates unsafe casting and map lookups.
"""

import re

from vulkan_codegen.models import FeatureMember, FeatureStruct
from vulkan_codegen.naming import (
    generate_getter_name,
    generate_member_name,
    get_short_feature_name,
    get_stype_from_name,
    to_camel_case,
    vk_name_to_cpp_name,
)
from vulkan_codegen.codegen import (
    build_extension_type_map,
    build_alias_maps,
    generate_file_header,
    propagate_ext_map_through_aliases,
)
from vulkan_codegen.spec import (
    VULKAN_SPEC_VERSION,
    build_skiplist,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)

out_path, include_path = get_output_paths()


def extract_feature_members(type_elem) -> tuple[str | None, list[FeatureMember]]:
    """Extract sType and VkBool32 members from a struct type element."""
    stype = None
    members = []

    for member in type_elem.findall("member"):
        member_type = member.find("type")
        member_name = member.find("name")

        if member_type is None or member_name is None:
            continue

        type_text = member_type.text
        name_text = member_name.text

        if name_text == "sType" and member.get("values"):
            stype = member.get("values")
        elif type_text == "VkBool32":
            comment = member.get("comment", "")
            members.append(FeatureMember(name=name_text, comment=comment))

    return stype, members


def find_vk_physical_device_features2(xml_root) -> FeatureStruct | None:
    """Find and create VkPhysicalDeviceFeatures2 struct."""
    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("name") == "VkPhysicalDeviceFeatures":
            vk_features_members = []
            for member in type_elem.findall("member"):
                member_type = member.find("type")
                member_name = member.find("name")
                if member_type is not None and member_name is not None:
                    if member_type.text == "VkBool32":
                        comment = member.get("comment", "")
                        vk_features_members.append(
                            FeatureMember(name=member_name.text, comment=comment)
                        )

            if vk_features_members:
                return FeatureStruct(
                    vk_name="VkPhysicalDeviceFeatures2",
                    cpp_name="PhysicalDeviceFeatures2",
                    stype="VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2",
                    members=vk_features_members,
                    member_prefix="features.",
                )
            break
    return None


def find_aliases_for_features(xml_root, features: list[FeatureStruct]):
    """Find and populate aliases for feature structures."""
    for type_elem in xml_root.findall("types/type"):
        alias = type_elem.get("alias")
        name = type_elem.get("name")
        if alias and name:
            for feat in features:
                if feat.vk_name == alias:
                    feat.aliases.append(name)
                    break


def find_feature_structures(xml_root, tags) -> list[FeatureStruct]:
    """Find all structures that extend VkPhysicalDeviceFeatures2."""
    features = []
    skiplist = build_skiplist(xml_root)
    extension_map = build_extension_type_map(xml_root)
    alias_to_canonical, _ = build_alias_maps(xml_root)
    propagate_ext_map_through_aliases(extension_map, alias_to_canonical)

    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("category") != "struct":
            continue

        struct_extends = type_elem.get("structextends", "")
        if "VkPhysicalDeviceFeatures2" not in struct_extends.split(","):
            continue

        vk_name = type_elem.get("name")
        if not vk_name or vk_name in skiplist or type_elem.get("alias") is not None:
            continue

        stype, members = extract_feature_members(type_elem)
        if not stype:
            stype = get_stype_from_name(vk_name)

        cpp_name = vk_name_to_cpp_name(vk_name)
        ext_info = extension_map.get(vk_name)
        extension = ext_info[0] if ext_info else None
        promotion_version = ext_info[2] if ext_info else None

        features.append(
            FeatureStruct(
                vk_name=vk_name,
                cpp_name=cpp_name,
                stype=stype,
                members=members,
                extension=extension,
                promotion_version=promotion_version,
            )
        )

    # Add VkPhysicalDeviceFeatures2
    if features2 := find_vk_physical_device_features2(xml_root):
        features.append(features2)

    find_aliases_for_features(xml_root, features)
    return features


def generate_concept_definition(features: list[FeatureStruct]) -> list[str]:
    """Generate VulkanFeatureStruct concept definition."""
    lines = [
        "/**",
        " * @brief C++20 concept for validating Vulkan feature structure types.",
        " * @tparam T The type to validate.",
        " */",
        "template <typename T>",
        "concept VulkanFeatureStruct = (",
    ]

    type_checks = [
        f"    std::same_as<T, vk::{feat.cpp_name}>"
        for feat in sorted(features, key=lambda f: f.cpp_name)
    ]
    lines.append(" ||\n".join(type_checks))
    lines.extend([");", ""])

    return lines


def generate_class_methods_declaration(features: list[FeatureStruct]) -> list[str]:
    """Generate method declarations for VulkanFeatures class."""
    lines = [
        "    /// Type-safe template access",
        "    template<typename T> requires VulkanFeatureStruct<T>",
        "    T& get();",
        "",
        "    template<typename T> requires VulkanFeatureStruct<T>",
        "    const T& get() const;",
        "",
        "    // Named getters for each feature struct",
    ]

    lines.append("    vk::PhysicalDeviceFeatures& get_features();")
    lines.append("    const vk::PhysicalDeviceFeatures& get_features() const;")
    lines.append("    operator const vk::PhysicalDeviceFeatures&() const;")
    for feat in sorted(features, key=lambda f: f.cpp_name):
        getter_name = generate_getter_name(feat.cpp_name)
        lines.append(f"    vk::{feat.cpp_name}& {getter_name}();")
        lines.append(f"    const vk::{feat.cpp_name}& {getter_name}() const;")
        lines.append(f"    operator const vk::{feat.cpp_name}&() const;")

    lines.extend(["", "    // Alias getters (for backwards compatibility)"])

    for feat in sorted(features, key=lambda f: f.cpp_name):
        for alias_vk_name in feat.aliases:
            alias_cpp_name = vk_name_to_cpp_name(alias_vk_name)
            alias_getter_name = generate_getter_name(alias_cpp_name)
            lines.append(
                f"    vk::{feat.cpp_name}& {alias_getter_name}();  // Alias for {feat.cpp_name}"
            )
            lines.append(f"    const vk::{feat.cpp_name}& {alias_getter_name}() const;")

    return lines


def generate_header(features: list[FeatureStruct]) -> str:
    """Generate the vulkan_features.hpp header file content."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
        "#pragma once",
        "",
        '#include "vulkan/vulkan.hpp"',
        "",
        "#include <cstdint>",
        "#include <memory>",
        "#include <string>",
        "#include <unordered_set>",
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "// Forward declarations",
        "class VulkanProperties;",
        "",
    ]

    lines.extend(generate_concept_definition(features))

    lines.extend(
        [
            "/**",
            " * @brief Aggregate class containing all Vulkan feature structures.",
            " *",
            " * Provides type-safe access to all ~240 Vulkan feature structs without",
            " * unsafe casting or map lookups. Each feature struct is stored as a direct member.",
            " *",
            " * Features can be accessed via:",
            " * - Template method: get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>()",
            " * - Named getter: get_ray_tracing_pipeline_features_khr()",
            ' * - String-based: set_feature("RayTracingPipelineKHR", "rayTracingPipeline", true)',
            " */",
            "class VulkanFeatures {",
            "  public:",
            "    /// Default constructor (zero-initialized features)",
            "    VulkanFeatures() = default;",
            "",
            "    /// Constructor: query features from physical device",
            "    VulkanFeatures(const vk::PhysicalDevice& physical_device,",
            "                   const VulkanProperties& properties);",
            "",
            '    /// Constructor: enable features from strings of the form "structName/featureName"',
            "    VulkanFeatures(const vk::ArrayProxy<const std::string>& features);",
            "",
            "    /// Copy and move",
            "    VulkanFeatures(const VulkanFeatures&) = default;",
            "    VulkanFeatures& operator=(const VulkanFeatures&) = default;",
            "    VulkanFeatures(VulkanFeatures&&) = default;",
            "    VulkanFeatures& operator=(VulkanFeatures&&) = default;",
            "",
        ]
    )

    lines.extend(generate_class_methods_declaration(features))

    lines.extend(
        [
            "",
            '    /// Enable features from strings of the form "structName/featureName"',
            "    void enable_features(const vk::ArrayProxy<const std::string>& features);",
            "",
            '    /// Disable features from strings of the form "structName/featureName"',
            "    void disable_features(const vk::ArrayProxy<const std::string>& features);",
            "",
            "    /// String-based feature access (runtime)",
            "    bool set_feature(const std::string& feature_struct_name,",
            "                     const std::string& feature_name,",
            "                     bool value);",
            "",
            "    bool get_feature(const std::string& feature_struct_name,",
            "                     const std::string& feature_name) const;",
            "",
            '    /// String-based feature access using "structName/featureName" format',
            "    void set_feature(const std::string& feature, bool value);",
            "",
            "    bool get_feature(const std::string& feature) const;",
            "",
            "    /// Get list of all feature names for a struct",
            "    std::vector<std::string> get_feature_names(const std::string& feature_struct_name) const;",
            "",
            '    /// Get list of all feature struct names (e.g., "RayTracingPipelineKHR", "16BitStorage")',
            "    /// Useful for iterating over all feature structs to merge or compare VulkanFeatures objects",
            "    std::vector<std::string> get_feature_struct_names() const;",
            "",
            "    /// Build pNext chain for device creation (includes only structs with non-zero features)",
            "    /// @param p_next Optional pointer to chain with other structures",
            "    void* build_chain_for_device_creation(void* p_next = nullptr);",
            "",
            "    /// Get all extensions required by currently enabled features",
            "    /// @return Vector of extension name macros needed for enabled features",
            "    std::vector<const char*> get_required_extensions() const;",
            "",
            "  private:",
            '    /// Internal: parse "structName/featureName" string into components',
            "    static std::pair<std::string, std::string> parse_feature_string(const std::string& feature);",
            "",
            "    /// Internal: parse and set features from strings",
            "    void parse_and_set_features(const vk::ArrayProxy<const std::string>& features, bool value);",
            "",
            "    /// Internal: get pointer to struct by StructureType",
            "    void* get_struct_ptr(vk::StructureType stype);",
            "    const void* get_struct_ptr(vk::StructureType stype) const;",
            "",
            "    // Feature struct members (~240 structs)",
        ]
    )

    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        lines.append(f"    vk::{feat.cpp_name} {member_name}{{}};")

    lines.extend(
        [
            "",
            "};",
            "",
            "/**",
            " * @brief Get the Vulkan structure type for a feature name.",
            ' * @param name The name of the feature structure (e.g., "PhysicalDeviceRobustness2FeaturesEXT").',
            " * @return The vk::StructureType, or std::nullopt if not found.",
            " */",
            "std::optional<vk::StructureType> structure_type_for_feature_name(const std::string& name);",
            "",
            "} // namespace merian",
            "",
        ]
    )

    return "\n".join(lines)


def generate_constructor(features: list[FeatureStruct], tags) -> list[str]:
    """Generate VulkanFeatures constructor implementation."""
    lines = [
        "VulkanFeatures::VulkanFeatures(const vk::PhysicalDevice& physical_device,",
        "                               const VulkanProperties& properties) {",
        "    // Enumerate device extensions to check support",
        "    std::unordered_set<std::string> device_extensions;",
        "    for (const auto& ext : physical_device.enumerateDeviceExtensionProperties()) {",
        "        device_extensions.insert(ext.extensionName);",
        "    }",
        "",
        "    // Get effective API version from properties",
        "    const uint32_t vk_api_version = properties.get_vk_api_version();",
        "",
        "    // Build pNext chain for all features with supported extensions",
        "    void* feat_p_next = nullptr;",
        "",
    ]

    sorted_features = sorted(
        [f for f in features if f.cpp_name != "PhysicalDeviceFeatures2"],
        key=lambda f: f.cpp_name,
    )

    for feat in sorted_features:
        member_name = generate_member_name(feat.cpp_name)
        lines.append(f"    // {feat.cpp_name}")

        if feat.extension and feat.promotion_version:
            lines.extend(
                [
                    f"    if (vk_api_version >= {feat.promotion_version} ||",
                    f"        device_extensions.contains({feat.extension})) {{",
                ]
            )
        elif feat.extension:
            lines.append(f"    if (device_extensions.contains({feat.extension})) {{")
        elif feat.promotion_version:
            lines.append(
                f"    if (vk_api_version >= {feat.promotion_version}) {{"
            )
        else:
            lines.append("    {")

        lines.extend(
            [
                f"        {member_name}.pNext = feat_p_next;",
                f"        feat_p_next = &{member_name};",
                "    }",
                "",
            ]
        )

    features2 = next(
        (f for f in features if f.cpp_name == "PhysicalDeviceFeatures2"), None
    )
    if features2:
        member_name = generate_member_name(features2.cpp_name)
        lines.extend(
            [
                "    // PhysicalDeviceFeatures2 (always at head of chain)",
                f"    {member_name}.pNext = feat_p_next;",
                "",
                "    // Query features from device",
                f"    physical_device.getFeatures2(&{member_name});",
            ]
        )

    lines.extend(["}", ""])
    return lines


def generate_template_get(features: list[FeatureStruct]) -> list[str]:
    """Generate template get<T>() methods and instantiations."""
    lines = [
        "// Template get<T>() implementation",
        "template<typename T> requires VulkanFeatureStruct<T>",
        "T& VulkanFeatures::get() {",
        "    return const_cast<T&>(static_cast<const VulkanFeatures*>(this)->get<T>());",
        "}",
        "",
        "template<typename T> requires VulkanFeatureStruct<T>",
        "const T& VulkanFeatures::get() const {",
        "    const auto stype = T::structureType;",
        "    const void* ptr = get_struct_ptr(stype);",
        "    if (!ptr) {",
        "        throw std::runtime_error(fmt::format(",
        '            "Unknown feature struct type: {}", vk::to_string(stype)));',
        "    }",
        "    return *reinterpret_cast<const T*>(ptr);",
        "}",
        "",
        "// Explicit template instantiations",
    ]

    for feat in sorted(features, key=lambda f: f.cpp_name):
        lines.extend(
            [
                f"template vk::{feat.cpp_name}& VulkanFeatures::get<vk::{feat.cpp_name}>();",
                f"template const vk::{feat.cpp_name}& VulkanFeatures::get<vk::{feat.cpp_name}>() const;",
            ]
        )

    lines.append("")
    return lines


def generate_named_getters(features: list[FeatureStruct]) -> list[str]:
    """Generate named getter implementations."""
    lines = ["// Named getter implementations"]

    lines.extend(
        [
            "vk::PhysicalDeviceFeatures& VulkanFeatures::get_features() {",
            "    return m_features2.features;",
            "}",
            "const vk::PhysicalDeviceFeatures& VulkanFeatures::get_features() const {",
            "    return m_features2.features;",
            "}",
            "VulkanFeatures::operator const vk::PhysicalDeviceFeatures&() const {",
            "    return m_features2.features;",
            "}",
        ]
    )

    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        getter_name = generate_getter_name(feat.cpp_name)
        lines.extend(
            [
                f"vk::{feat.cpp_name}& VulkanFeatures::{getter_name}() {{",
                f"    return {member_name};",
                "}",
                f"const vk::{feat.cpp_name}& VulkanFeatures::{getter_name}() const {{",
                f"    return {member_name};",
                "}",
                f"VulkanFeatures::operator const vk::{feat.cpp_name}&() const {{",
                f"    return {member_name};",
                "}",
            ]
        )

    lines.append("")
    return lines


def generate_alias_getters(features: list[FeatureStruct]) -> list[str]:
    """Generate alias getter implementations."""
    lines = ["// Alias getter implementations (for backwards compatibility)"]

    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        for alias_vk_name in feat.aliases:
            alias_cpp_name = vk_name_to_cpp_name(alias_vk_name)
            alias_getter_name = generate_getter_name(alias_cpp_name)
            lines.extend(
                [
                    f"vk::{feat.cpp_name}& VulkanFeatures::{alias_getter_name}() {{",
                    f"    return {member_name};  // Alias for {feat.cpp_name}",
                    "}",
                    f"const vk::{feat.cpp_name}& VulkanFeatures::{alias_getter_name}() const {{",
                    f"    return {member_name};",
                    "}",
                ]
            )

    lines.append("")
    return lines


def generate_get_struct_ptr(features: list[FeatureStruct], tags) -> list[str]:
    """Generate get_struct_ptr switch implementations."""
    lines = [
        "void* VulkanFeatures::get_struct_ptr(vk::StructureType stype) {",
        "    return const_cast<void*>(static_cast<const VulkanFeatures*>(this)->get_struct_ptr(stype));",
        "}",
        "",
        "const void* VulkanFeatures::get_struct_ptr(vk::StructureType stype) const {",
        "    switch (stype) {",
    ]

    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
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


def generate_set_get_feature(features: list[FeatureStruct], tags) -> list[str]:
    """Generate set_feature and get_feature implementations."""
    lines = [
        "bool VulkanFeatures::set_feature(const std::string& feature_struct_name,",
        "                                  const std::string& feature_name,",
        "                                  bool value) {",
        "    auto stype_opt = structure_type_for_feature_name(feature_struct_name);",
        "    if (!stype_opt) return false;",
        "    ",
        "    const vk::StructureType stype = *stype_opt;",
        "    const vk::Bool32 vk_value = value ? VK_TRUE : VK_FALSE;",
        "    ",
        "    switch (stype) {",
    ]

    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)
        prefix = feat.member_prefix

        lines.append(f"        case vk::StructureType::{stype_camel}: {{")
        for member in feat.members:
            lines.append(
                f'            if (feature_name == "{member.name}") {{ {member_name}.{prefix}{member.name} = vk_value; return true; }}'
            )
        lines.extend(["            return false;", "        }"])

    lines.extend(
        [
            "        default:",
            "            return false;",
            "    }",
            "}",
            "",
            "bool VulkanFeatures::get_feature(const std::string& feature_struct_name,",
            "                                  const std::string& feature_name) const {",
            "    auto stype_opt = structure_type_for_feature_name(feature_struct_name);",
            "    if (!stype_opt) return false;",
            "    ",
            "    const vk::StructureType stype = *stype_opt;",
            "    ",
            "    switch (stype) {",
        ]
    )

    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)
        prefix = feat.member_prefix

        lines.append(f"        case vk::StructureType::{stype_camel}: {{")
        for member in feat.members:
            lines.append(
                f'            if (feature_name == "{member.name}") return {member_name}.{prefix}{member.name} == VK_TRUE;'
            )
        lines.extend(["            return false;", "        }"])

    lines.extend(
        [
            "        default:",
            "            return false;",
            "    }",
            "}",
            "",
        ]
    )

    return lines


def generate_helper_functions(features: list[FeatureStruct], tags) -> list[str]:
    """Generate get_feature_names, get_feature_struct_names, build_chain, get_required_extensions."""
    lines = [
        "std::vector<std::string> VulkanFeatures::get_feature_names(const std::string& feature_struct_name) const {",
        "    auto stype_opt = structure_type_for_feature_name(feature_struct_name);",
        "    if (!stype_opt) return {};",
        "    ",
        "    const vk::StructureType stype = *stype_opt;",
        "    ",
        "    switch (stype) {",
    ]

    for feat in sorted(features, key=lambda f: f.cpp_name):
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)
        lines.append(f"        case vk::StructureType::{stype_camel}:")
        lines.append("            return {")
        for member in feat.members:
            lines.append(f'                "{member.name}",')
        lines.append("            };")

    lines.extend(
        [
            "        default:",
            "            return {};",
            "    }",
            "}",
            "",
            "std::vector<std::string> VulkanFeatures::get_feature_struct_names() const {",
            "    return {",
        ]
    )

    for feat in sorted(features, key=lambda f: f.cpp_name):
        short_name = get_short_feature_name(feat.cpp_name, tags)
        lines.append(f'        "{short_name}",')

    lines.extend(
        [
            "    };",
            "}",
            "",
            "void* VulkanFeatures::build_chain_for_device_creation(void* p_next) {",
            "    void* chain_head = p_next;",
            "    ",
            "    // Build chain from all structs with at least one enabled feature",
        ]
    )

    for feat in reversed(sorted(features, key=lambda f: f.cpp_name)):
        if not feat.members:
            continue

        member_name = generate_member_name(feat.cpp_name)
        prefix = feat.member_prefix
        conditions = [
            f"{member_name}.{prefix}{m.name} == VK_TRUE" for m in feat.members
        ]
        condition_str = " ||\n        ".join(conditions)

        lines.extend(
            [
                f"    // Check {feat.cpp_name}",
                f"    if ({condition_str}) {{",
                f"        {member_name}.pNext = chain_head;",
                f"        chain_head = &{member_name};",
                "    }",
                "",
            ]
        )

    lines.extend(
        [
            "    return chain_head;",
            "}",
            "",
            "std::vector<const char*> VulkanFeatures::get_required_extensions() const {",
            "    std::unordered_set<const char*> extensions;",
            "    ",
        ]
    )

    for feat in sorted(features, key=lambda f: f.cpp_name):
        if not feat.extension or not feat.members:
            continue

        member_name = generate_member_name(feat.cpp_name)
        prefix = feat.member_prefix
        conditions = [
            f"{member_name}.{prefix}{m.name} == VK_TRUE" for m in feat.members
        ]
        condition_str = " || ".join(conditions)

        lines.extend(
            [
                f"    if ({condition_str}) {{",
                f"        extensions.insert({feat.extension});",
                "    }",
            ]
        )

    lines.extend(
        [
            "    ",
            "    return std::vector<const char*>(extensions.begin(), extensions.end());",
            "}",
            "",
        ]
    )

    return lines


def generate_structure_type_lookup(features: list[FeatureStruct], tags) -> list[str]:
    """Generate structure_type_for_feature_name function."""
    lines = [
        "std::optional<vk::StructureType> structure_type_for_feature_name(const std::string& name) {",
        "    static const std::unordered_map<std::string, vk::StructureType> name_map = {",
    ]

    for feat in sorted(features, key=lambda f: f.cpp_name):
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)
        lines.append(
            f'        {{"{feat.cpp_name}", vk::StructureType::{stype_camel}}},'
        )

        short_name = get_short_feature_name(feat.cpp_name, tags)
        if short_name != feat.cpp_name:
            lines.append(
                f'        {{"{short_name}", vk::StructureType::{stype_camel}}},'
            )

    lines.extend(
        [
            "    };",
            "    ",
            "    auto it = name_map.find(name);",
            "    if (it != name_map.end()) {",
            "        return it->second;",
            "    }",
            "    return std::nullopt;",
            "}",
            "",
        ]
    )

    return lines


def generate_string_features_methods() -> list[str]:
    """Generate parse_and_set_features, string constructor, enable/disable methods."""
    return [
        "std::pair<std::string, std::string> VulkanFeatures::parse_feature_string(const std::string& feature) {",
        "    const auto sep = feature.find('/');",
        "    if (sep == std::string::npos) {",
        "        throw std::invalid_argument(fmt::format(",
        "            \"Invalid feature string '{}': expected 'structName/featureName'\", feature));",
        "    }",
        "    return {feature.substr(0, sep), feature.substr(sep + 1)};",
        "}",
        "",
        "void VulkanFeatures::parse_and_set_features(const vk::ArrayProxy<const std::string>& features, bool value) {",
        "    for (const auto& feature : features) {",
        "        const auto [struct_name, feature_name] = parse_feature_string(feature);",
        "        if (!set_feature(struct_name, feature_name, value)) {",
        "            throw std::invalid_argument(fmt::format(",
        "                \"Unknown feature '{}' in struct '{}'\", feature_name, struct_name));",
        "        }",
        "    }",
        "}",
        "",
        "VulkanFeatures::VulkanFeatures(const vk::ArrayProxy<const std::string>& features) {",
        "    parse_and_set_features(features, true);",
        "}",
        "",
        "void VulkanFeatures::enable_features(const vk::ArrayProxy<const std::string>& features) {",
        "    parse_and_set_features(features, true);",
        "}",
        "",
        "void VulkanFeatures::disable_features(const vk::ArrayProxy<const std::string>& features) {",
        "    parse_and_set_features(features, false);",
        "}",
        "",
        "void VulkanFeatures::set_feature(const std::string& feature, bool value) {",
        "    const auto [struct_name, feature_name] = parse_feature_string(feature);",
        "    if (!set_feature(struct_name, feature_name, value)) {",
        "        throw std::invalid_argument(fmt::format(",
        "            \"Unknown feature '{}' in struct '{}'\", feature_name, struct_name));",
        "    }",
        "}",
        "",
        "bool VulkanFeatures::get_feature(const std::string& feature) const {",
        "    const auto [struct_name, feature_name] = parse_feature_string(feature);",
        "    return get_feature(struct_name, feature_name);",
        "}",
        "",
    ]


def generate_implementation(features: list[FeatureStruct], tags) -> str:
    """Generate the vulkan_features.cpp implementation file content."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
        '#include "merian/vk/utils/vulkan_features.hpp"',
        '#include "merian/vk/utils/vulkan_properties.hpp"',
        "",
        "#include <fmt/format.h>",
        "#include <stdexcept>",
        "#include <unordered_map>",
        "",
        "namespace merian {",
        "",
    ]

    lines.extend(generate_constructor(features, tags))
    lines.extend(generate_string_features_methods())
    lines.extend(generate_template_get(features))
    lines.extend(generate_named_getters(features))
    lines.extend(generate_alias_getters(features))
    lines.extend(generate_get_struct_ptr(features, tags))
    lines.extend(generate_set_get_feature(features, tags))
    lines.extend(generate_helper_functions(features, tags))
    lines.extend(generate_structure_type_lookup(features, tags))

    lines.extend(["} // namespace merian", ""])

    return "\n".join(lines)


def main():
    xml_root = load_vulkan_spec()
    tags = load_vendor_tags(xml_root)

    print("Finding feature structures...")
    features = find_feature_structures(xml_root, tags)
    print(f"Found {len(features)} feature structures")

    print("\nExample features found:")
    for feat in features[:5]:
        print(f"  - {feat.cpp_name}: {len(feat.members)} members")
        if feat.extension:
            print(f"    Extension: {feat.extension}")
        if feat.members:
            print(f"    Members: {', '.join(m.name for m in feat.members[:3])}...")

    print(f"\nGenerating header file: {include_path / 'vulkan_features.hpp'}")
    header_content = generate_header(features)
    with open(include_path / "vulkan_features.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'vulkan_features.cpp'}")
    impl_content = generate_implementation(features, tags)
    with open(out_path / "vulkan_features.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Generated {len(features)} feature struct members")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
