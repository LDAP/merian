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

from pprint import pprint
from vulkan_codegen.codegen import (
    build_alias_maps,
    build_extension_type_map,
    build_feature_version_map,
    generate_file_header,
)
from vulkan_codegen.models import FeatureMember, FeatureStruct
from vulkan_codegen.naming import (
    generate_getter_name,
    generate_member_name,
    get_short_feature_name,
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
    # Extension map now handles aliases internally during building
    extension_map = build_extension_type_map(xml_root)
    version_map = build_feature_version_map(xml_root)

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
        required_version = version_map.get(vk_name)

        features.append(
            FeatureStruct(
                vk_name=vk_name,
                cpp_name=cpp_name,
                stype=stype,
                members=members,
                extension=extension,
                promotion_version=promotion_version,
                required_version=required_version,
                structextends=struct_extends.split(",") if struct_extends else [],
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
        "class PhysicalDevice;",
        "using PhysicalDeviceHandle = std::shared_ptr<PhysicalDevice>;",
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
            ' * - Feature name only: set_feature("robustImageAccess", true)',
            " * - Template method: get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>()",
            " * - Named getter: get_ray_tracing_pipeline_features_khr()",
            " *",
            " * Version-portable: Promoted features (e.g., robustImageAccess) are set in ALL",
            " * matching structs (both core and extension). The constructor's pNext chain builder",
            " * selects the appropriate struct based on device API version and extension support.",
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
            "    /// Constructor: enable features from feature names",
            "    VulkanFeatures(const vk::ArrayProxy<const std::string>& feature_names);",
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
            "    /// Feature-name-only API (simplified, version-portable)",
            "    void set_feature(const std::string& feature_name, bool enable);",
            "    bool get_feature(const std::string& feature_name) const;",
            "",
            "    /// Enable multiple features by name",
            "    void enable_features(const vk::ArrayProxy<const std::string>& feature_names);",
            "",
            "    /// Get list of all available feature names",
            "    std::vector<std::string> get_all_feature_names() const;",
            "",
            "    /// Get list of all currently enabled feature names",
            "    std::vector<std::string> get_enabled_features() const;",
            "",
            "    /// Build pNext chain for device creation (includes only supported structs with enabled features)",
            "    /// @param physical_device The physical device to query for API version and extension support",
            "    /// @param p_next Optional pointer to chain with other structures",
            "    void* build_chain_for_device_creation(",
            "        const PhysicalDeviceHandle& physical_device,",
            "        void* p_next = nullptr);",
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
        "    // PHASE 1: Build pNext chain for all features with supported extensions/versions",
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

        # Check extension + promotion first before required_version alone
        # This handles promoted structs that have all three attributes
        if feat.extension and feat.promotion_version:
            lines.extend(
                [
                    f"    if (vk_api_version >= {feat.promotion_version} ||",
                    f"        device_extensions.contains({feat.extension})) {{",
                ]
            )
        elif feat.required_version:
            # Core feature struct - requires specific API version (VulkanXXFeatures)
            lines.append(f"    if (vk_api_version >= {feat.required_version}) {{")
        elif feat.extension:
            lines.append(f"    if (device_extensions.contains({feat.extension})) {{")
        elif feat.promotion_version:
            lines.append(f"    if (vk_api_version >= {feat.promotion_version}) {{")
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
                "    // PhysicalDeviceFeatures2",
                f"    {member_name}.pNext = feat_p_next;",
                "",
                "    // Query features from device",
                f"    physical_device.getFeatures2(&{member_name});",
                "",
            ]
        )

    # PHASE 2: Sync features between structs that share feature names
    lines.extend(
        [
            "    // PHASE 2: Sync features between structs",
            "    // containing the same feature (due to promotion / deprecation).",
            "    // Example: bufferDeviceAddress appears in KHR, EXT, and Vulkan12Features",
            "",
        ]
    )

    # Build feature-to-structs map to find duplicates
    feature_to_structs = {}
    for feat in features:
        for member in feat.members:
            if member.name not in feature_to_structs:
                feature_to_structs[member.name] = []
            feature_to_structs[member.name].append((feat, member))

    # For each feature that appears in multiple structs, generate OR-based sync code
    synced_any_feature = False
    for feature_name, struct_members in sorted(feature_to_structs.items()):
        if len(struct_members) <= 1:
            continue  # Feature only appears in one struct, no sync needed

        synced_any_feature = True
        lines.append(
            f"    // Sync {feature_name} with all {len(struct_members)} structs that provide the feature."
        )

        # Build OR expression of all struct values
        or_parts = []
        for feat, member in struct_members:
            member_name = generate_member_name(feat.cpp_name)
            prefix = feat.member_prefix
            or_parts.append(f"{member_name}.{prefix}{member.name}")

        or_expression = " || ".join(or_parts)

        # Store OR result in temporary variable, then set in all structs
        lines.append(f"    const VkBool32 {feature_name}_value = {or_expression};")

        # Set the computed value in all structs
        for feat, member in struct_members:
            member_name = generate_member_name(feat.cpp_name)
            prefix = feat.member_prefix
            lines.append(
                f"    {member_name}.{prefix}{member.name} = {feature_name}_value;"
            )

        lines.append("")

    if not synced_any_feature:
        lines.append("    // No duplicate features found, no syncing needed")
        lines.append("")

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


def build_feature_to_structs_map(features: list[FeatureStruct]) -> dict:
    """Build mapping of feature name -> list of (FeatureStruct, member) tuples."""
    feature_map = {}
    for feat in features:
        for member in feat.members:
            if member.name not in feature_map:
                feature_map[member.name] = []
            feature_map[member.name].append({"struct": feat, "member": member})
    return feature_map


def generate_set_feature(feature_map: dict) -> list[str]:
    """Generate set_feature implementation with type-safe if/else chain."""
    lines = [
        "void VulkanFeatures::set_feature(const std::string& feature_name, bool enable) {",
        "    const VkBool32 value = enable ? VK_TRUE : VK_FALSE;",
        "",
    ]

    first = True
    for feature_name, struct_list in sorted(feature_map.items()):
        if_keyword = "if" if first else "else if"
        first = False

        lines.append(f'    {if_keyword} (feature_name == "{feature_name}") {{')

        # Set in ALL structs that contain this feature
        for entry in struct_list:
            member_name = generate_member_name(entry["struct"].cpp_name)
            prefix = entry["struct"].member_prefix
            feat_name = entry["member"].name
            lines.append(f"        {member_name}.{prefix}{feat_name} = value;")

        lines.append("    }")

    lines.extend(
        [
            "    else {",
            "        throw std::invalid_argument(",
            "            fmt::format(\"Unknown feature: '{}'\", feature_name));",
            "    }",
            "}",
            "",
        ]
    )

    return lines


def generate_get_feature(feature_map: dict) -> list[str]:
    """Generate get_feature implementation with type-safe if/else chain."""
    lines = [
        "bool VulkanFeatures::get_feature(const std::string& feature_name) const {",
    ]

    first = True
    for feature_name, struct_list in sorted(feature_map.items()):
        if_keyword = "if" if first else "else if"
        first = False

        # Return from first struct (all aliases must be consistent per Vulkan spec)
        entry = struct_list[0]
        member_name = generate_member_name(entry["struct"].cpp_name)
        prefix = entry["struct"].member_prefix
        feat_name = entry["member"].name

        lines.append(f'    {if_keyword} (feature_name == "{feature_name}") {{')
        lines.append(f"        return {member_name}.{prefix}{feat_name} == VK_TRUE;")
        lines.append("    }")

    lines.extend(
        [
            "    else {",
            "        throw std::invalid_argument(",
            "            fmt::format(\"Unknown feature: '{}'\", feature_name));",
            "    }",
            "}",
            "",
        ]
    )

    return lines


def generate_helper_functions(features: list[FeatureStruct], tags) -> list[str]:
    """Generate build_chain_for_device_creation and get_required_extensions."""
    lines = [
        "void* VulkanFeatures::build_chain_for_device_creation(",
        "    const PhysicalDeviceHandle& physical_device,",
        "    void* p_next) {",
        "    const uint32_t vk_api_version = physical_device->get_vk_api_version();",
        "    void* chain_head = p_next;",
        "    ",
        "    // Build chain from supported structs with at least one enabled feature",
        "    // Prefers VulkanXXFeatures over individual extension structs when BOTH are available",
        "",
    ]

    # Build map: promotion_version -> aggregate struct's required_version
    # Aggregate structs are those with required_version but NO extension
    # (they collect features from multiple promoted extensions)
    # This tells us when aggregate feature structs became available
    aggregate_struct_availability = {}
    for feat in features:
        if feat.required_version and not feat.extension:
            # This is an aggregate struct (like PhysicalDeviceVulkan11Features)
            # It becomes available at its required_version
            # Map from its promotion_version (if it has one) to its required_version
            if feat.promotion_version:
                aggregate_struct_availability[feat.promotion_version] = (
                    feat.required_version
                )

    # Sort: Aggregate structs first (will be added to chain first due to reverse iteration)
    # Aggregate structs = have required_version but no extension
    sorted_features = sorted(
        features,
        key=lambda f: (not (f.required_version and not f.extension), f.cpp_name),
    )

    for feat in reversed(sorted_features):
        if not feat.members:
            continue

        # Skip non-extending structs - only include structs that extend VkPhysicalDeviceFeatures2
        # Base features (VkPhysicalDeviceFeatures) should be accessed via get_features() instead
        # PhysicalDeviceFeatures2 extends VkDeviceCreateInfo but not itself, so it's filtered out
        if "VkPhysicalDeviceFeatures2" not in feat.structextends:
            continue

        member_name = generate_member_name(feat.cpp_name)
        prefix = feat.member_prefix

        # Build feature enable condition
        feature_conditions = [
            f"{member_name}.{prefix}{m.name} == VK_TRUE" for m in feat.members
        ]
        feature_condition_str = " ||\n        ".join(feature_conditions)

        lines.append(f"    // Check {feat.cpp_name}")

        # For extension structs that have promoted features:
        # - Only use if corresponding aggregate struct is NOT supported by device
        # - This ensures aggregate structs are preferred when available
        if feat.promotion_version and feat.extension:
            # This is an extension struct with promoted features
            # Check if an aggregate struct is available for this promotion version
            aggregate_version = aggregate_struct_availability.get(
                feat.promotion_version
            )

            if aggregate_version:
                lines.append(f"    if ({feature_condition_str}) {{")
                lines.append(
                    f"        // Extension struct for features promoted to {feat.promotion_version}"
                )
                lines.append(
                    f"        // Only include if aggregate struct not available (requires >= {aggregate_version})"
                )
                lines.append(
                    f"        assert((physical_device->extension_supported({feat.extension}) ||"
                )
                lines.append(
                    f"                vk_api_version >= {feat.promotion_version}) &&"
                )
                lines.append(
                    '               "Feature enabled but neither extension nor promoted version supported");'
                )
                lines.append(f"        if (vk_api_version < {aggregate_version}) {{")
                lines.append(
                    "            // Aggregate struct not supported, use extension struct"
                )
                lines.append(f"            {member_name}.pNext = chain_head;")
                lines.append(f"            chain_head = &{member_name};")
                lines.append("        } else {")
                lines.append(
                    "            // Aggregate struct is supported, it will be used instead"
                )
                lines.append("        }")
                lines.append("    }")
                lines.append("")
                continue

        # Build support condition
        # Check extension + promotion first, before checking required_version alone
        # This handles structs that have ALL THREE attributes
        if feat.extension and feat.promotion_version:
            # Promoted extension feature struct (may also have required_version)
            lines.append(f"    if ({feature_condition_str}) {{")
            lines.append(
                f"        assert((physical_device->extension_supported({feat.extension}) ||"
            )
            lines.append(
                f"                vk_api_version >= {feat.promotion_version}) &&"
            )
            lines.append(
                '               "Feature enabled but neither extension nor promoted version supported");'
            )
        elif feat.required_version and not feat.extension:
            # Aggregate struct (core-only, no extension fallback)
            # Check version FIRST, then features
            # Features might be enabled via syncing from extension structs,
            # but we can't use this aggregate struct if the device doesn't support the required version
            lines.append(f"    if (vk_api_version >= {feat.required_version}) {{")
            lines.append(f"        if ({feature_condition_str}) {{")
            lines.append(f"            {member_name}.pNext = chain_head;")
            lines.append(f"            chain_head = &{member_name};")
            lines.append("        }")
            lines.append("    }")
            lines.append("")
            continue
        elif feat.extension:
            # Extension-only feature - assert extension is supported when feature is enabled
            lines.append(f"    if ({feature_condition_str}) {{")
            lines.append(
                f"        assert(physical_device->extension_supported({feat.extension}) &&"
            )
            lines.append(
                '               "Feature enabled but required extension not supported");'
            )
        elif feat.promotion_version:
            # Core VulkanXXFeatures struct - requires API version
            lines.append(f"    if ({feature_condition_str}) {{")
            lines.append(
                f"        assert(vk_api_version >= {feat.promotion_version} &&"
            )
            lines.append(
                '               "Feature enabled but required Vulkan version not supported");'
            )
        else:
            # Base Vulkan 1.0 features (no version/extension requirement)
            lines.append(f"    if ({feature_condition_str}) {{")

        lines.extend(
            [
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


def generate_string_features_methods(feature_map: dict) -> list[str]:
    """Generate constructor, enable_features, and feature query methods."""
    lines = [
        "VulkanFeatures::VulkanFeatures(const vk::ArrayProxy<const std::string>& feature_names) {",
        "    enable_features(feature_names);",
        "}",
        "",
        "void VulkanFeatures::enable_features(const vk::ArrayProxy<const std::string>& feature_names) {",
        "    for (const auto& feature_name : feature_names) {",
        "        set_feature(feature_name, true);",
        "    }",
        "}",
        "",
        "std::vector<std::string> VulkanFeatures::get_all_feature_names() const {",
        "    return {",
    ]

    # Add all feature names
    for feature_name in sorted(feature_map.keys()):
        lines.append(f'        "{feature_name}",')

    lines.extend(
        [
            "    };",
            "}",
            "",
            "std::vector<std::string> VulkanFeatures::get_enabled_features() const {",
            "    std::vector<std::string> enabled;",
            "",
        ]
    )

    # Check each feature by directly accessing struct members (not calling get_feature)
    for feature_name, struct_list in sorted(feature_map.items()):
        # Use first struct to check (all aliases must be consistent per Vulkan spec)
        entry = struct_list[0]
        member_name = generate_member_name(entry["struct"].cpp_name)
        prefix = entry["struct"].member_prefix
        feat_name = entry["member"].name

        lines.append(
            f'    if ({member_name}.{prefix}{feat_name} == VK_TRUE) enabled.push_back("{feature_name}");'
        )

    lines.extend(
        [
            "",
            "    return enabled;",
            "}",
            "",
        ]
    )

    return lines


def generate_implementation(features: list[FeatureStruct], tags) -> str:
    """Generate the vulkan_features.cpp implementation file content."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
        '#include "merian/vk/utils/vulkan_features.hpp"',
        "",
        '#include "merian/vk/physical_device.hpp"',
        '#include "merian/vk/utils/vulkan_properties.hpp"',
        "",
        "#include <cassert>",
        "#include <fmt/format.h>",
        "#include <stdexcept>",
        "#include <unordered_map>",
        "",
        "namespace merian {",
        "",
    ]

    # Build feature-to-structs mapping for new API
    feature_map = build_feature_to_structs_map(features)

    lines.extend(generate_constructor(features, tags))
    lines.extend(generate_string_features_methods(feature_map))
    lines.extend(generate_template_get(features))
    lines.extend(generate_named_getters(features))
    lines.extend(generate_alias_getters(features))
    lines.extend(generate_get_struct_ptr(features, tags))
    lines.extend(generate_set_feature(feature_map))
    lines.extend(generate_get_feature(feature_map))
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
