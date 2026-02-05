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
from pprint import pprint
from vulkan_codegen.codegen import (
    generate_file_header,
    generate_stype_switch,
    get_extension,
)
from vulkan_codegen.models import FeatureMember, FeatureStruct
from vulkan_codegen.parsing import build_extension_map, find_all_structures
from vulkan_codegen.naming import (
    generate_getter_name,
    generate_member_name,
    get_short_feature_name,
    get_stype_from_name,
    to_camel_case,
    vk_name_to_cpp_name,
)
from vulkan_codegen.spec import (
    FEATURE_STRUCT_BASE,
    DEVICE_CREATE_INFO,
    PHYSICAL_DEVICE_PREFIX,
    VULKAN_SPEC_VERSION,
    build_skiplist,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)

out_path, include_path = get_output_paths()

# Global extension map for accessing extension properties
_extension_map = {}


def _extract_feature_members(vulkan_struct, struct_map: dict) -> tuple[list[FeatureMember], str]:
    """
    Extract VkBool32 feature members from VulkanStruct.

    For VkPhysicalDeviceFeatures2, recursively looks up VkPhysicalDeviceFeatures
    and extracts its VkBool32 members with the appropriate prefix.

    Returns:
        (feature_members, feature_member_prefix)
    """
    feature_members = []
    feature_member_prefix = ""

    # Check if this struct has VkPhysicalDeviceFeatures member (Features2 case)
    for member_type, member_name, member_comment in vulkan_struct.members:
        if member_type == "VkPhysicalDeviceFeatures":
            # Recursive lookup: get features from VkPhysicalDeviceFeatures
            feature_member_prefix = f"{member_name}."  # e.g., "features."
            if base_struct := struct_map.get("VkPhysicalDeviceFeatures"):
                for base_type, base_name, base_comment in base_struct.members:
                    if base_type == "VkBool32":
                        feature_members.append(FeatureMember(name=base_name, comment=base_comment))
            return feature_members, feature_member_prefix

    # Regular case: extract VkBool32 members directly
    for member_type, member_name, member_comment in vulkan_struct.members:
        if member_type == "VkBool32":
            feature_members.append(FeatureMember(name=member_name, comment=member_comment))

    return feature_members, feature_member_prefix


def find_feature_structures(xml_root, tags) -> list[FeatureStruct]:
    """Find all structures that extend VkPhysicalDeviceFeatures2 or VkDeviceCreateInfo."""
    skiplist = build_skiplist(xml_root)

    # Build full extension map
    global _extension_map
    _extension_map = build_extension_map(xml_root, tags)

    # Extract all structs using unified function
    all_structs = find_all_structures(xml_root, tags, skiplist)

    # Filter for feature structs (includes base struct and all extensions)
    feature_structs = [
        s for s in all_structs
        if s.vk_name == FEATURE_STRUCT_BASE  # Include VkPhysicalDeviceFeatures2 itself
        or FEATURE_STRUCT_BASE in s.structextends  # Or structs that extend it
    ]

    # Build lookup map for recursive member resolution
    struct_map = {s.vk_name: s for s in all_structs}

    features = []
    for vulkan_struct in feature_structs:
        # Extract VkBool32 members and determine prefix
        feature_members, feature_member_prefix = _extract_feature_members(vulkan_struct, struct_map)

        # Get required_version from version map for core structs
        from vulkan_codegen.codegen import build_feature_version_map
        version_map = build_feature_version_map(xml_root)
        required_version = version_map.get(vulkan_struct.vk_name)

        features.append(FeatureStruct(
            vk_name=vulkan_struct.vk_name,
            cpp_name=vulkan_struct.cpp_name,
            stype=vulkan_struct.stype,
            extension_name=vulkan_struct.extension_name,
            structextends=vulkan_struct.structextends,
            aliases=vulkan_struct.aliases,
            is_alias=vulkan_struct.is_alias,
            members=vulkan_struct.members,  # Keep raw members from base class
            feature_members=feature_members,  # Filtered VkBool32 members
            required_version=required_version,
            feature_member_prefix=feature_member_prefix,  # From recursive lookup
        ))

    return features


def _canonical_features_only(features: list[FeatureStruct]) -> list[FeatureStruct]:
    """Filter out aliases - only return canonical structs for member/getter generation."""
    return [f for f in features if not f.is_alias]


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

    # Only include canonical structs (no aliases) in concept
    canonical = _canonical_features_only(features)
    type_checks = [
        f"    std::same_as<T, vk::{feat.cpp_name}>"
        for feat in sorted(canonical, key=lambda f: f.cpp_name)
    ]
    lines.append(" ||\n".join(type_checks))
    lines.extend([");", ""])

    return lines


def generate_class_methods_declaration(features: list[FeatureStruct]) -> list[str]:
    """Generate method declarations for VulkanFeatures class."""
    lines = [
        "    /// Type-safe template access (const only - use set_feature for modifications)",
        "    template<typename T> requires VulkanFeatureStruct<T>",
        "    const T& get() const;",
        "",
        "    // Named getters for each feature struct",
    ]

    # Only canonical structs get main getters
    canonical = _canonical_features_only(features)

    lines.append("    const vk::PhysicalDeviceFeatures& get_features() const;")
    lines.append("    operator const vk::PhysicalDeviceFeatures&() const;")
    for feat in sorted(canonical, key=lambda f: f.cpp_name):
        getter_name = generate_getter_name(feat.cpp_name)
        lines.append(f"    const vk::{feat.cpp_name}& {getter_name}() const;")
        lines.append(f"    operator const vk::{feat.cpp_name}&() const;")

    lines.extend(["", "    // Alias getters (for backwards compatibility)"])

    # Generate alias getters from canonical struct's aliases list
    for feat in sorted(canonical, key=lambda f: f.cpp_name):
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
            "    bool set_feature(const std::string& feature_name, bool enable);",
            "    bool get_feature(const std::string& feature_name) const;",
            "",
            "    /// Enable multiple features by name",
            "    void enable_features(const vk::ArrayProxy<const std::string>& feature_names);",
            "",
            "    /// Get list of all available feature names",
            "    static const std::vector<std::string>& get_all_feature_names();",
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

    # Only canonical structs get member declarations (no aliases)
    canonical = _canonical_features_only(features)
    for feat in sorted(canonical, key=lambda f: f.cpp_name):
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
        "    // PHASE 1: Query ALL supported feature structs to populate their values",
        "    // Both VkPhysicalDeviceVulkan{XX}Features and individual promoted structs are queried.",
        "    // Device creation will prefer VulkanXXFeatures when available (see build_chain_for_device_creation).",
        "    void* feat_p_next = nullptr;",
        "",
    ]

    # Only canonical features (no aliases) that extend the base struct
    canonical = _canonical_features_only(features)
    extending_features = [f for f in canonical if FEATURE_STRUCT_BASE in f.structextends]
    sorted_features = sorted(extending_features, key=lambda f: f.cpp_name)

    for feat in sorted_features:
        member_name = generate_member_name(feat.cpp_name)
        lines.append(f"    // {feat.cpp_name}")

        # Build condition: promotion_version (when promoted to core) OR extension
        # For VulkanXXFeatures structs, use required_version since they have no extension
        conditions = []

        # Get extension info if this struct is from an extension
        ext = get_extension(feat, _extension_map)

        # Prefer promotion_version over required_version to avoid duplicates
        # (e.g., a struct promoted in 1.2 shouldn't check both required_version=1.2 and promotion_version=1.2)
        if ext and ext.promotedto:
            # Struct was promoted from an extension - check promotion version OR extension
            conditions.append(f"vk_api_version >= {ext.promotedto}")
            conditions.append(f"device_extensions.contains({ext.name_macro})")
        elif feat.required_version:
            # Core struct (like VulkanXXFeatures) - only check required version
            conditions.append(f"vk_api_version >= {feat.required_version}")
        elif ext:
            # Extension-only struct - check extension
            conditions.append(f"device_extensions.contains({ext.name_macro})")

        if conditions:
            condition = " || ".join(conditions)
            lines.extend([
                f"    if ({condition}) {{",
                f"        {member_name}.pNext = feat_p_next;",
                f"        feat_p_next = &{member_name};",
                "    }",
                "",
            ])
        else:
            # No conditions - always available
            lines.extend([
                f"    {member_name}.pNext = feat_p_next;",
                f"    feat_p_next = &{member_name};",
                "",
            ])

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

    # Build feature-to-structs map to find duplicates (canonical only)
    canonical = _canonical_features_only(features)
    feature_to_structs = {}
    for feat in canonical:
        for member in feat.feature_members:
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
            prefix = feat.feature_member_prefix
            or_parts.append(f"{member_name}.{prefix}{member.name}")

        or_expression = " || ".join(or_parts)

        # Store OR result in temporary variable, then set in all structs
        lines.append(f"    const VkBool32 {feature_name}_value = {or_expression};")

        # Set the computed value in all structs
        for feat, member in struct_members:
            member_name = generate_member_name(feat.cpp_name)
            prefix = feat.feature_member_prefix
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
    """Generate template get<T>() const method and instantiations."""
    lines = [
        "// Template get<T>() implementation (const only - use set_feature for modifications)",
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

    # Only canonical structs get template instantiations (no aliases)
    canonical = _canonical_features_only(features)
    for feat in sorted(canonical, key=lambda f: f.cpp_name):
        lines.append(f"template const vk::{feat.cpp_name}& VulkanFeatures::get<vk::{feat.cpp_name}>() const;")

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

    # Only canonical structs get main getters (no aliases)
    canonical = _canonical_features_only(features)
    for feat in sorted(canonical, key=lambda f: f.cpp_name):
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

    # Only canonical structs have aliases to generate getters for
    canonical = _canonical_features_only(features)
    for feat in sorted(canonical, key=lambda f: f.cpp_name):
        canonical_getter_name = generate_getter_name(feat.cpp_name)
        for alias_vk_name in feat.aliases:
            alias_cpp_name = vk_name_to_cpp_name(alias_vk_name)
            alias_getter_name = generate_getter_name(alias_cpp_name)
            lines.extend(
                [
                    f"vk::{feat.cpp_name}& VulkanFeatures::{alias_getter_name}() {{",
                    f"    return {canonical_getter_name}();  // Alias for {feat.cpp_name}",
                    "}",
                    f"const vk::{feat.cpp_name}& VulkanFeatures::{alias_getter_name}() const {{",
                    f"    return {canonical_getter_name}();  // Alias for {feat.cpp_name}",
                    "}",
                ]
            )

    lines.append("")
    return lines


def generate_get_struct_ptr(features: list[FeatureStruct], tags) -> list[str]:
    """Generate get_struct_ptr const switch implementation."""
    # Only canonical features (no aliases) - aliases share sType with canonical
    canonical = _canonical_features_only(features)
    return generate_stype_switch(canonical, "VulkanFeatures", tags)


def build_feature_to_structs_map(features: list[FeatureStruct]) -> dict:
    """Build mapping of feature name -> list of (FeatureStruct, member) tuples."""
    # Only canonical features (no aliases)
    canonical = _canonical_features_only(features)
    feature_map = {}
    for feat in canonical:
        for member in feat.feature_members:
            if member.name not in feature_map:
                feature_map[member.name] = []
            feature_map[member.name].append({"struct": feat, "member": member})
    return feature_map


def generate_set_feature(feature_map: dict) -> list[str]:
    """Generate set_feature implementation with type-safe if/else chain."""
    lines = [
        "bool VulkanFeatures::set_feature(const std::string& feature_name, bool enable) {",
        "    const VkBool32 value = enable ? VK_TRUE : VK_FALSE;",
        "",
    ]

    for feature_name, struct_list in sorted(feature_map.items()):
        lines.append(f'    if (feature_name == "{feature_name}") {{')

        # Set in ALL structs that contain this feature
        for entry in struct_list:
            member_name = generate_member_name(entry["struct"].cpp_name)
            prefix = entry["struct"].feature_member_prefix
            feat_name = entry["member"].name
            lines.append(f"        {member_name}.{prefix}{feat_name} = value;")

        lines.append(f"        return true;")
        lines.append("    }")

    lines.extend(
        [
            "    ",
            "    return false;",
            "    ",
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
        prefix = entry["struct"].feature_member_prefix
        feat_name = entry["member"].name

        lines.append(f'    {if_keyword} (feature_name == "{feature_name}") {{')
        lines.append(f"        return {member_name}.{prefix}{feat_name} == VK_TRUE;")
        lines.append("    }")

    lines.extend(
        [
            "    else {",
            "        return false;",
            "    }",
            "}",
            "",
        ]
    )

    return lines


def compute_feature_fingerprint(feat: FeatureStruct) -> frozenset[str]:
    """Compute a fingerprint (set of member names) for a feature struct."""
    return frozenset(m.name for m in feat.feature_members)


def compute_priority_score(feat: FeatureStruct) -> tuple:
    """Compute priority score for sorting. Higher = better priority.

    1. Supersets (more features) come first
    2. Then prefer non-deprecated over deprecated structs
    3. Then prefer newer extensions (higher extension number)

    Note: Uses global _extension_map for extension lookup.
    """
    # Look up extension properties
    is_deprecated = 0
    extension_number = 0

    ext = get_extension(feat, _extension_map)
    if ext:
        is_deprecated = 1 if ext.deprecatedby else 0
        extension_number = ext.extension_number or 0

    return (
        len(feat.feature_members),  # More features = superset = higher priority
        -is_deprecated,  # Non-deprecated (0) > deprecated (1), so negate for sorting
        extension_number,  # Higher extension number = newer
        feat.vk_name,  # Stable tie-breaker
    )


def group_features_by_relationship(features: list[FeatureStruct]) -> list:
    """Group features into nested hierarchies.

    Returns nested structure where each group can be:
    - A single feature (independent)
    - A list [parent, ...children] where children can themselves be groups

    Children with identical features are grouped as aliases.
    """
    # Compute fingerprints for all features
    feat_fingerprints = [compute_feature_fingerprint(feat) for feat in features]

    def build_hierarchy(candidates: list[int], assigned: set[int]) -> list:
        """Recursively build hierarchy from candidate indices."""
        groups = []

        for i in candidates:
            if i in assigned:
                continue

            parent_fp = feat_fingerprints[i]
            if not parent_fp:
                continue

            # Find direct children (proper subsets)
            children_indices = []
            for j in candidates:
                if i == j or j in assigned:
                    continue
                child_fp = feat_fingerprints[j]
                if not child_fp:
                    continue
                if child_fp < parent_fp:  # Proper subset
                    children_indices.append(j)

            if children_indices:
                # Mark children as assigned
                for j in children_indices:
                    assigned.add(j)

                # Group children by their feature fingerprints (find aliases)
                child_groups_by_fp = {}
                for j in children_indices:
                    fp = feat_fingerprints[j]
                    if fp not in child_groups_by_fp:
                        child_groups_by_fp[fp] = []
                    child_groups_by_fp[fp].append(j)

                # Build child groups (aliases or further hierarchies)
                child_groups = []
                for fp, indices in child_groups_by_fp.items():
                    if len(indices) > 1:
                        # Aliases - sort by priority
                        alias_group = [features[idx] for idx in indices]
                        alias_group.sort(key=compute_priority_score, reverse=True)
                        child_groups.append(alias_group)
                    else:
                        # Single child - might have its own children
                        # Recursively build its hierarchy
                        child_hierarchy = build_hierarchy([indices[0]], assigned)
                        if child_hierarchy:
                            child_groups.extend(child_hierarchy)
                        else:
                            child_groups.append(features[indices[0]])

                # Create parent-child group
                group = [features[i]] + child_groups
                groups.append(group)
                assigned.add(i)

        return groups

    # Start with all features as candidates
    assigned = set()
    all_indices = list(range(len(features)))
    hierarchies = build_hierarchy(all_indices, assigned)

    return hierarchies


def generate_feature_enabled_condition(feat: FeatureStruct) -> str:
    """Generate condition string for checking if any features are enabled."""
    if not feat.feature_members:
        return "false"

    member_name = generate_member_name(feat.cpp_name)
    prefix = feat.feature_member_prefix
    conditions = [f"{member_name}.{prefix}{m.name} == VK_TRUE" for m in feat.feature_members]
    return " ||\n        ".join(conditions)


def generate_support_assertion(feat: FeatureStruct) -> list[str]:
    """Generate assertion for extension/version support."""
    lines = []

    ext = get_extension(feat, _extension_map)

    if ext and ext.promotedto:
        lines.extend([
            f"        assert((physical_device->extension_supported({ext.name_macro}) ||",
            f"                vk_api_version >= {ext.promotedto}) &&",
            f'               "Feature enabled but neither extension nor promoted version supported");',
        ])
    elif ext:
        lines.extend([
            f"        assert(physical_device->extension_supported({ext.name_macro}) &&",
            f'               "Feature enabled but required extension not supported");',
        ])
    elif feat.required_version:
        lines.extend([
            f"        assert(vk_api_version >= {feat.required_version} &&",
            f'               "Feature enabled but required Vulkan version not supported");',
        ])

    return lines


def generate_device_creation_for_item(item, indent: str = "    ") -> list[str]:
    """Recursively generate code for a feature or nested group.

    item can be:
    - A FeatureStruct (leaf node)
    - A list [parent, ...children] (hierarchy)
    """
    if isinstance(item, FeatureStruct):
        # Leaf node - generate simple if block
        return generate_single_feature(item, indent)
    elif isinstance(item, list) and len(item) > 0:
        # Check if this is aliases or parent-child
        if all(isinstance(x, FeatureStruct) for x in item):
            # All FeatureStructs - check if aliases
            fingerprints = {compute_feature_fingerprint(f) for f in item}
            if len(fingerprints) == 1:
                # Aliases - generate if/else-if chain
                return generate_aliases(item, indent)
            else:
                # Parent-child where children are all leaves
                return generate_parent_child(item[0], item[1:], indent)
        else:
            # Mixed - first is parent, rest are nested groups
            return generate_parent_child(item[0], item[1:], indent)
    return []


def generate_single_feature(feat: FeatureStruct, indent: str) -> list[str]:
    """Generate simple if block for a single feature."""
    lines = []
    member_name = generate_member_name(feat.cpp_name)
    feature_condition = generate_feature_enabled_condition(feat)

    ext = get_extension(feat, _extension_map)

    support_conditions = []
    if ext and ext.promotedto:
        support_conditions.append(f"vk_api_version >= {ext.promotedto}")
    if ext:
        support_conditions.append(f"physical_device->extension_supported({ext.name_macro})")
    support_condition = " || ".join(support_conditions) if support_conditions else "true"

    lines.extend([
        f"{indent}if (({feature_condition}) &&",
        f"{indent}    ({support_condition})) {{",
    ])
    lines.extend([indent + "    " + line for line in generate_support_assertion(feat)])
    lines.extend([
        f"{indent}    {member_name}.pNext = chain_head;",
        f"{indent}    chain_head = &{member_name};",
        f"{indent}}}",
    ])
    return lines


def generate_aliases(aliases: list[FeatureStruct], indent: str) -> list[str]:
    """Generate if/else-if chain for aliases (same features)."""
    lines = []
    for i, feat in enumerate(aliases):
        member_name = generate_member_name(feat.cpp_name)
        feature_condition = generate_feature_enabled_condition(feat)
        if_keyword = "if" if i == 0 else "else if"

        ext = get_extension(feat, _extension_map)

        support_conditions = []
        if feat.required_version:
            support_conditions.append(f"vk_api_version >= {feat.required_version}")
        elif ext and ext.promotedto:
            support_conditions.append(f"vk_api_version >= {ext.promotedto}")
        if ext:
            support_conditions.append(f"physical_device->extension_supported({ext.name_macro})")
        support_condition = " || ".join(support_conditions) if support_conditions else "true"

        lines.extend([
            f"{indent}{if_keyword} (({feature_condition}) &&",
            f"{indent}    ({support_condition})) {{",
        ])
        lines.extend([indent + "    " + line for line in generate_support_assertion(feat)])
        lines.extend([
            f"{indent}    {member_name}.pNext = chain_head;",
            f"{indent}    chain_head = &{member_name};",
            f"{indent}}}",
        ])
    return lines


def generate_parent_child(parent: FeatureStruct, children: list, indent: str) -> list[str]:
    """Generate if (parent) {...} else { children } with recursive nesting."""
    lines = []

    # Parent condition
    parent_member = generate_member_name(parent.cpp_name)
    parent_feature_cond = generate_feature_enabled_condition(parent)

    parent_ext = get_extension(parent, _extension_map)

    parent_support_cond = []
    if parent.required_version:
        parent_support_cond.append(f"vk_api_version >= {parent.required_version}")
    elif parent_ext and parent_ext.promotedto:
        parent_support_cond.append(f"vk_api_version >= {parent_ext.promotedto}")
    if parent_ext:
        parent_support_cond.append(f"physical_device->extension_supported({parent_ext.name_macro})")
    parent_support = " || ".join(parent_support_cond) if parent_support_cond else "true"

    lines.extend([
        f"{indent}if (({parent_feature_cond}) &&",
        f"{indent}    ({parent_support})) {{",
    ])
    lines.extend([indent + "    " + line for line in generate_support_assertion(parent)])
    lines.extend([
        f"{indent}    {parent_member}.pNext = chain_head;",
        f"{indent}    chain_head = &{parent_member};",
        f"{indent}}} else {{",
    ])

    # Recursively generate children
    for child in children:
        child_lines = generate_device_creation_for_item(child, indent + "    ")
        lines.extend(child_lines)

    lines.append(f"{indent}}}")
    return lines


def count_features_recursive(item) -> int:
    """Count total FeatureStructs in a (possibly nested) group."""
    if isinstance(item, FeatureStruct):
        return 1
    elif isinstance(item, list):
        return sum(count_features_recursive(x) for x in item)
    return 0


def generate_device_creation_for_group(group, group_idx: int) -> list[str]:
    """Generate code for a hierarchical group with comment."""
    lines = []

    # Add comment
    if isinstance(group, list) and len(group) > 0 and isinstance(group[0], FeatureStruct):
        total = count_features_recursive(group)
        lines.append(f"    // Group {group_idx}: {group[0].cpp_name} and {total-1} related struct(s)")

    # Generate recursive code
    item_lines = generate_device_creation_for_item(group)
    lines.extend(item_lines)
    lines.append("")

    return lines


def generate_device_creation_for_independent(feat: FeatureStruct) -> list[str]:
    """Generate simple if block for an independent struct (not in any group)."""
    lines = []
    member_name = generate_member_name(feat.cpp_name)
    feature_condition = generate_feature_enabled_condition(feat)

    lines.extend([
        f"    // {feat.cpp_name}",
        f"    if ({feature_condition}) {{",
    ])

    # Add assertion
    lines.extend(["    " + line for line in generate_support_assertion(feat)])

    # Add to chain
    lines.extend([
        f"        {member_name}.pNext = chain_head;",
        f"        chain_head = &{member_name};",
        "    }",
        "",
    ])

    return lines


def generate_helper_functions(features: list[FeatureStruct], tags) -> list[str]:
    """Generate build_chain_for_device_creation and get_required_extensions."""
    lines = [
        "void* VulkanFeatures::build_chain_for_device_creation(",
        "    const PhysicalDeviceHandle& physical_device,",
        "    void* p_next) {",
        "    const uint32_t vk_api_version = physical_device->get_vk_api_version();",
        "    void* chain_head = p_next;",
        "",
        "    // Build chain using priority-based grouping",
        "    // For structs with related features (same/subset/superset), use if/else chains",
        "    // Priority: higher API version > promoted > newer extension number",
        "    // This ensures mutual exclusivity and prefers newer/better versions",
        "",
    ]

    # Filter to only canonical features that extend VkPhysicalDeviceFeatures2 and have members (no aliases)
    canonical = _canonical_features_only(features)
    relevant_features = [
        f for f in canonical
        if f.feature_members and DEVICE_CREATE_INFO in f.structextends
    ]

    # Group features by their relationships
    groups = group_features_by_relationship(relevant_features)

    # Get features that are in groups (to exclude from independent list)
    def extract_all_features(item, result: set):
        """Recursively extract all FeatureStructs from nested hierarchy."""
        if isinstance(item, FeatureStruct):
            result.add(item.vk_name)
        elif isinstance(item, list):
            for x in item:
                extract_all_features(x, result)

    grouped_features = set()
    for group in groups:
        extract_all_features(group, grouped_features)

    # Generate if/else chains for each group
    lines.append("    // Feature groups (related structs with priority ordering)")
    lines.append("")
    for i, group in enumerate(groups, 1):
        lines.extend(generate_device_creation_for_group(group, i))

    # Generate simple if blocks for independent features
    independent = [
        f for f in sorted(relevant_features, key=lambda f: f.cpp_name)
        if f.vk_name not in grouped_features
    ]

    if independent:
        lines.append("    // Independent features (no aliases or aggregation relationships)")
        lines.append("")
        for feat in independent:
            lines.extend(generate_device_creation_for_independent(feat))

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

    # Only canonical structs (no aliases) for extension requirements
    canonical = _canonical_features_only(features)
    for feat in sorted(canonical, key=lambda f: f.cpp_name):
        ext = get_extension(feat, _extension_map)
        if not ext or not feat.feature_members:
            continue

        member_name = generate_member_name(feat.cpp_name)
        prefix = feat.feature_member_prefix
        conditions = [
            f"{member_name}.{prefix}{m.name} == VK_TRUE" for m in feat.feature_members
        ]
        condition_str = " || ".join(conditions)

        lines.extend(
            [
                f"    if ({condition_str}) {{",
                f"        extensions.insert({ext.name_macro});",
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

    # Only canonical structs (no aliases)
    canonical = _canonical_features_only(features)
    for feat in sorted(canonical, key=lambda f: f.cpp_name):
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
        "const std::vector<std::string>& VulkanFeatures::get_all_feature_names() {",
        "    static const std::vector<std::string> feature_names = {",
    ]

    # Add all feature names
    for feature_name in sorted(feature_map.keys()):
        lines.append(f'        "{feature_name}",')

    lines.extend(
        [
            "    };",
            "    return feature_names;",
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
        prefix = entry["struct"].feature_member_prefix
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
