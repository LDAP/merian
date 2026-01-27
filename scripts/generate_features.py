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

import datetime
import re

from vulkan_codegen.models import FeatureMember, FeatureStruct
from vulkan_codegen.naming import (
    generate_getter_name,
    generate_member_name,
    get_stype_from_name,
    to_camel_case,
    vk_name_to_cpp_name,
)
from vulkan_codegen.spec import (
    VULKAN_SPEC_VERSION,
    build_skiplist,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)

out_path, include_path = get_output_paths()


def get_short_feature_name(cpp_name: str, tags: list[str]) -> str:
    """
    Generate short feature name from cpp_name.
    E.g., "PhysicalDeviceRayTracingPipelineFeaturesKHR" -> "RayTracingPipelineKHR"
    """
    short_name = cpp_name

    for tag in tags:
        short_name = short_name.removesuffix(tag)

    short_name = short_name.removesuffix("Features")
    short_name = short_name.removeprefix("PhysicalDevice")

    if cpp_name == "PhysicalDeviceFeatures2":
        short_name = "Vulkan10"

    return short_name


def find_feature_structures(xml_root, tags) -> list[FeatureStruct]:
    """
    Find all structures that extend VkPhysicalDeviceFeatures2.
    These are the feature structures we want to generate classes for.
    """
    features = []

    skiplist = build_skiplist(xml_root)

    # Build a map of extension names to their required extensions and promotion versions
    extension_map = {}

    def parse_promotion_version(depends: str) -> str | None:
        """Extract VK_VERSION from depends attribute."""
        if not depends:
            return None
        match = re.search(r"VK_VERSION_(\d+)_(\d+)", depends)
        if match:
            major, minor = match.groups()
            return f"VK_API_VERSION_{major}_{minor}"
        return None

    # Find extension definitions
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

        promotedto = ext.get("promotedto", "")
        promotion_version = parse_promotion_version(promotedto) if promotedto else None

        for req in ext.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if type_name:
                    extension_map[type_name] = (ext_name_macro, promotion_version)

    # Add aliases to extension_map
    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("category") != "struct":
            continue

        alias_of = type_elem.get("alias")
        type_name = type_elem.get("name")

        if alias_of and type_name:
            if alias_of in extension_map:
                extension_map[type_name] = extension_map[alias_of]
            elif type_name in extension_map:
                extension_map[alias_of] = extension_map[type_name]

    # Find all structures that extend VkPhysicalDeviceFeatures2
    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("category") != "struct":
            continue

        struct_extends = type_elem.get("structextends", "")
        if "VkPhysicalDeviceFeatures2" not in struct_extends.split(","):
            continue

        vk_name = type_elem.get("name")
        if not vk_name:
            continue

        if vk_name in skiplist:
            continue

        if type_elem.get("alias") is not None:
            continue

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

        if not stype:
            stype = get_stype_from_name(vk_name)

        cpp_name = vk_name_to_cpp_name(vk_name)

        ext_info = extension_map.get(vk_name)
        extension = ext_info[0] if ext_info else None
        promotion_version = ext_info[1] if ext_info else None

        feature = FeatureStruct(
            vk_name=vk_name,
            cpp_name=cpp_name,
            stype=stype,
            members=members,
            extension=extension,
            promotion_version=promotion_version,
        )

        features.append(feature)

    # Add VkPhysicalDeviceFeatures2
    vk_features_members = []
    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("name") == "VkPhysicalDeviceFeatures":
            for member in type_elem.findall("member"):
                member_type = member.find("type")
                member_name = member.find("name")
                if member_type is not None and member_name is not None:
                    if member_type.text == "VkBool32":
                        comment = member.get("comment", "")
                        vk_features_members.append(
                            FeatureMember(name=member_name.text, comment=comment)
                        )
            break

    if vk_features_members:
        features.append(
            FeatureStruct(
                vk_name="VkPhysicalDeviceFeatures2",
                cpp_name="PhysicalDeviceFeatures2",
                stype="VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2",
                members=vk_features_members,
                member_prefix="features.",
            )
        )

    # Find aliases
    for type_elem in xml_root.findall("types/type"):
        alias = type_elem.get("alias")
        name = type_elem.get("name")
        if alias and name:
            for feat in features:
                if feat.vk_name == alias:
                    feat.aliases.append(name)
                    break

    return features


def generate_header(features: list[FeatureStruct]) -> str:
    """Generate the vulkan_features.hpp header file content."""
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
        "#include <unordered_set>",
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "// Forward declaration",
        "class Instance;",
        "using InstanceHandle = std::shared_ptr<Instance>;",
        "",
    ]

    # Generate VulkanFeatureStruct concept
    lines.append("/**")
    lines.append(
        " * @brief C++20 concept for validating Vulkan feature structure types."
    )
    lines.append(" * @tparam T The type to validate.")
    lines.append(" */")
    lines.append("template <typename T>")
    lines.append("concept VulkanFeatureStruct = (")

    type_checks = []
    for feat in sorted(features, key=lambda f: f.cpp_name):
        type_checks.append(f"    std::same_as<T, vk::{feat.cpp_name}>")

    lines.append(" ||\n".join(type_checks))
    lines.append(");")
    lines.append("")

    # Generate VulkanFeatures class
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
            "                   const InstanceHandle& instance);",
            "",
            "    /// Copy and move",
            "    VulkanFeatures(const VulkanFeatures&) = default;",
            "    VulkanFeatures& operator=(const VulkanFeatures&) = default;",
            "    VulkanFeatures(VulkanFeatures&&) = default;",
            "    VulkanFeatures& operator=(VulkanFeatures&&) = default;",
            "",
            "    /// Type-safe template access",
            "    template<typename T> requires VulkanFeatureStruct<T>",
            "    T& get();",
            "",
            "    template<typename T> requires VulkanFeatureStruct<T>",
            "    const T& get() const;",
            "",
        ]
    )

    # Generate named getters
    lines.append("    // Named getters for each feature struct")
    for feat in sorted(features, key=lambda f: f.cpp_name):
        getter_name = generate_getter_name(feat.cpp_name)
        lines.append(f"    vk::{feat.cpp_name}& {getter_name}();")
        lines.append(f"    const vk::{feat.cpp_name}& {getter_name}() const;")

    # Generate alias getters
    lines.append("")
    lines.append("    // Alias getters (for backwards compatibility)")
    for feat in sorted(features, key=lambda f: f.cpp_name):
        for alias_vk_name in feat.aliases:
            alias_cpp_name = vk_name_to_cpp_name(alias_vk_name)
            alias_getter_name = generate_getter_name(alias_cpp_name)
            lines.append(f"    vk::{feat.cpp_name}& {alias_getter_name}();  // Alias for {feat.cpp_name}")
            lines.append(f"    const vk::{feat.cpp_name}& {alias_getter_name}() const;")

    lines.extend(
        [
            "",
            "    /// String-based feature access (runtime)",
            "    bool set_feature(const std::string& feature_struct_name,",
            "                     const std::string& feature_name,",
            "                     bool value);",
            "",
            "    bool get_feature(const std::string& feature_struct_name,",
            "                     const std::string& feature_name) const;",
            "",
            "    /// Get list of all feature names for a struct",
            "    std::vector<std::string> get_feature_names(const std::string& feature_struct_name) const;",
            "",
            "    /// Get list of all feature struct names (e.g., \"RayTracingPipelineKHR\", \"16BitStorage\")",
            "    /// Useful for iterating over all feature structs to merge or compare VulkanFeatures objects",
            "    std::vector<std::string> get_feature_struct_names() const;",
            "",
            "    /// Build pNext chain for device creation (includes only structs with non-zero features)",
            "    void* build_chain_for_device_creation();",
            "",
            "    /// Get all extensions required by currently enabled features",
            "    /// @return Vector of extension name macros needed for enabled features",
            "    std::vector<const char*> get_required_extensions() const;",
            "",
            "  private:",
            "    /// Internal: get pointer to struct by StructureType",
            "    void* get_struct_ptr(vk::StructureType stype);",
            "    const void* get_struct_ptr(vk::StructureType stype) const;",
            "",
        ]
    )

    # Generate member variables
    lines.append("    // Feature struct members (~240 structs)")
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


def generate_implementation(features: list[FeatureStruct], tags) -> str:
    """Generate the vulkan_features.cpp implementation file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        '#include "merian/vk/utils/vulkan_features.hpp"',
        '#include "merian/vk/instance.hpp"',
        "",
        "#include <fmt/format.h>",
        "#include <stdexcept>",
        "#include <unordered_map>",
        "",
        "namespace merian {",
        "",
    ]

    # Generate constructor
    lines.extend(
        [
            "VulkanFeatures::VulkanFeatures(const vk::PhysicalDevice& physical_device,",
            "                               const InstanceHandle& instance) {",
            "    // Enumerate device extensions to check support",
            "    std::unordered_set<std::string> device_extensions;",
            "    for (const auto& ext : physical_device.enumerateDeviceExtensionProperties()) {",
            "        device_extensions.insert(ext.extensionName);",
            "    }",
            "",
            "    // Build pNext chain for all features with supported extensions",
            "    void* feat_p_next = nullptr;",
            "",
        ]
    )

    # Sort features - PhysicalDeviceFeatures2 must be last (head of chain)
    sorted_features = sorted(
        [f for f in features if f.cpp_name != "PhysicalDeviceFeatures2"],
        key=lambda f: f.cpp_name,
    )
    features2 = next(
        (f for f in features if f.cpp_name == "PhysicalDeviceFeatures2"), None
    )

    # Generate chain building for each feature
    for feat in sorted_features:
        member_name = generate_member_name(feat.cpp_name)
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)

        lines.append(f"    // {feat.cpp_name}")
        if feat.extension and feat.promotion_version:
            lines.append(
                f"    if (instance->get_vk_api_version() >= {feat.promotion_version} ||"
            )
            lines.append(f"        device_extensions.contains({feat.extension})) {{")
        elif feat.extension:
            lines.append(f"    if (device_extensions.contains({feat.extension})) {{")
        elif feat.promotion_version:
            lines.append(
                f"    if (instance->get_vk_api_version() >= {feat.promotion_version}) {{"
            )
        else:
            lines.append("    {")

        lines.append(f"        {member_name}.pNext = feat_p_next;")
        lines.append(f"        feat_p_next = &{member_name};")
        lines.append("    }")
        lines.append("")

    # Add PhysicalDeviceFeatures2 at the head
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

    lines.extend(
        [
            "}",
            "",
        ]
    )

    # Generate template get<T>() methods
    lines.extend(
        [
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
        ]
    )

    # Generate explicit template instantiations
    lines.append("// Explicit template instantiations")
    for feat in sorted(features, key=lambda f: f.cpp_name):
        lines.append(
            f"template vk::{feat.cpp_name}& VulkanFeatures::get<vk::{feat.cpp_name}>();"
        )
        lines.append(
            f"template const vk::{feat.cpp_name}& VulkanFeatures::get<vk::{feat.cpp_name}>() const;"
        )
    lines.append("")

    # Generate named getters
    lines.append("// Named getter implementations")
    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        getter_name = generate_getter_name(feat.cpp_name)
        lines.append(f"vk::{feat.cpp_name}& VulkanFeatures::{getter_name}() {{")
        lines.append(f"    return {member_name};")
        lines.append("}")
        lines.append(
            f"const vk::{feat.cpp_name}& VulkanFeatures::{getter_name}() const {{"
        )
        lines.append(f"    return {member_name};")
        lines.append("}")

    lines.append("")

    # Generate alias getter implementations
    lines.append("// Alias getter implementations (for backwards compatibility)")
    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        for alias_vk_name in feat.aliases:
            alias_cpp_name = vk_name_to_cpp_name(alias_vk_name)
            alias_getter_name = generate_getter_name(alias_cpp_name)
            lines.append(f"vk::{feat.cpp_name}& VulkanFeatures::{alias_getter_name}() {{")
            lines.append(f"    return {member_name};  // Alias for {feat.cpp_name}")
            lines.append("}")
            lines.append(f"const vk::{feat.cpp_name}& VulkanFeatures::{alias_getter_name}() const {{")
            lines.append(f"    return {member_name};")
            lines.append("}")

    lines.append("")

    # Generate get_struct_ptr
    lines.extend(
        [
            "void* VulkanFeatures::get_struct_ptr(vk::StructureType stype) {",
            "    return const_cast<void*>(static_cast<const VulkanFeatures*>(this)->get_struct_ptr(stype));",
            "}",
            "",
            "const void* VulkanFeatures::get_struct_ptr(vk::StructureType stype) const {",
            "    switch (stype) {",
        ]
    )

    for feat in sorted(features, key=lambda f: f.cpp_name):
        member_name = generate_member_name(feat.cpp_name)
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum, tags)
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

    # Generate set_feature and get_feature
    lines.extend(
        [
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
    )

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
        lines.append("            return false;")
        lines.append("        }")

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
        lines.append("            return false;")
        lines.append("        }")

    lines.extend(
        [
            "        default:",
            "            return false;",
            "    }",
            "}",
            "",
        ]
    )

    # Generate get_feature_names
    lines.extend(
        [
            "std::vector<std::string> VulkanFeatures::get_feature_names(const std::string& feature_struct_name) const {",
            "    auto stype_opt = structure_type_for_feature_name(feature_struct_name);",
            "    if (!stype_opt) return {};",
            "    ",
            "    const vk::StructureType stype = *stype_opt;",
            "    ",
            "    switch (stype) {",
        ]
    )

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
        ]
    )

    # Generate get_feature_struct_names
    lines.extend(
        [
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
        ]
    )

    # Generate build_chain_for_device_creation
    lines.extend(
        [
            "void* VulkanFeatures::build_chain_for_device_creation() {",
            "    void* chain_head = nullptr;",
            "    ",
            "    // Build chain from all structs with at least one enabled feature",
        ]
    )

    for feat in reversed(sorted(features, key=lambda f: f.cpp_name)):
        member_name = generate_member_name(feat.cpp_name)
        prefix = feat.member_prefix

        if feat.members:
            lines.append(f"    // Check {feat.cpp_name}")
            conditions = [
                f"{member_name}.{prefix}{m.name} == VK_TRUE" for m in feat.members
            ]
            condition_str = " ||\n        ".join(conditions)
            lines.append(f"    if ({condition_str}) {{")
            lines.append(f"        {member_name}.pNext = chain_head;")
            lines.append(f"        chain_head = &{member_name};")
            lines.append("    }")
            lines.append("")

    lines.extend(
        [
            "    return chain_head;",
            "}",
            "",
        ]
    )

    # Generate get_required_extensions
    lines.extend(
        [
            "std::vector<const char*> VulkanFeatures::get_required_extensions() const {",
            "    std::unordered_set<const char*> extensions;",
            "    ",
        ]
    )

    for feat in sorted(features, key=lambda f: f.cpp_name):
        if not feat.extension:
            continue

        member_name = generate_member_name(feat.cpp_name)
        prefix = feat.member_prefix

        if feat.members:
            conditions = [
                f"{member_name}.{prefix}{m.name} == VK_TRUE" for m in feat.members
            ]
            condition_str = " || ".join(conditions)
            lines.append(f"    if ({condition_str}) {{")
            lines.append(f"        extensions.insert({feat.extension});")
            lines.append("    }")

    lines.extend(
        [
            "    ",
            "    return std::vector<const char*>(extensions.begin(), extensions.end());",
            "}",
            "",
        ]
    )

    # Generate structure_type_for_feature_name
    lines.extend(
        [
            "std::optional<vk::StructureType> structure_type_for_feature_name(const std::string& name) {",
            "    static const std::unordered_map<std::string, vk::StructureType> name_map = {",
        ]
    )

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
            "} // namespace merian",
            "",
        ]
    )

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
