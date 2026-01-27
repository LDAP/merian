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


@dataclass
class FeatureMember:
    """Represents a VkBool32 member in a feature structure."""

    name: str
    comment: str = ""


@dataclass
class FeatureStruct:
    """Represents a Vulkan feature structure like VkPhysicalDeviceRobustness2FeaturesEXT."""

    vk_name: str  # e.g., VkPhysicalDeviceRobustness2FeaturesEXT
    cpp_name: str  # e.g., PhysicalDeviceRobustness2FeaturesEXT
    stype: str  # e.g., VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT
    members: list[FeatureMember] = field(default_factory=list)
    extension: Optional[str] = None  # e.g., VK_EXT_ROBUSTNESS_2_EXTENSION_NAME
    promotion_version: Optional[str] = (
        None  # e.g., VK_API_VERSION_1_2 (when extension became core)
    )
    member_prefix: str = ""  # e.g., "features." for VkPhysicalDeviceFeatures2
    aliases: list[str] = field(default_factory=list)


def vk_name_to_cpp_name(vk_name: str) -> str:
    """Convert VkPhysicalDeviceFoo to PhysicalDeviceFoo."""
    return vk_name.removeprefix("Vk")


def get_stype_from_name(name: str) -> str:
    """
    Convert a struct name to its sType enum value.
    e.g., VkPhysicalDeviceRobustness2FeaturesEXT -> VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT
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


def get_short_feature_name(cpp_name: str, tags: list[str]) -> str:
    """
    Generate short feature name from cpp_name.
    E.g., "PhysicalDeviceRayTracingPipelineFeaturesKHR" -> "RayTracingPipelineKHR"
    """
    short_name = cpp_name

    # Remove vendor tags
    for tag in tags:
        short_name = short_name.removesuffix(tag)

    # Remove "Features" suffix
    short_name = short_name.removesuffix("Features")

    # Remove "PhysicalDevice" prefix
    short_name = short_name.removeprefix("PhysicalDevice")

    # Special case for Vulkan 1.0 features
    if cpp_name == "PhysicalDeviceFeatures2":
        short_name = "Vulkan10"

    return short_name


def parse_vulkan_spec():
    """Download and parse the Vulkan XML specification."""
    global tags
    print(f"Downloading Vulkan spec {VULKAN_SPEC_VERSION}...")
    with urlopen(VULKAN_SPEC_URL) as response:
        xml_root = ET.parse(response).getroot()

    # Load vendor tags for proper casing
    tags = [i.get("name") for i in xml_root.findall("tags/tag")]  # pyright: ignore[reportAssignmentType]
    return xml_root


def find_feature_structures(xml_root) -> list[FeatureStruct]:
    """
    Find all structures that extend VkPhysicalDeviceFeatures2.
    These are the feature structures we want to generate classes for.
    """
    features = []

    # Build skiplist for platform-specific types (same logic as generate_enum_string_map.py)
    skiplist = set()

    for ext in xml_root.findall("extensions/extension"):
        # Skip types from platform-specific extensions
        if ext.get("platform") is not None:
            for req in ext.findall("require"):
                for type_elem in req.findall("type"):
                    if name := type_elem.get("name"):
                        skiplist.add(name)

        # Skip types from extensions not supported on vulkan
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

    # Build a map of extension names to their required extensions and promotion versions
    extension_map = {}  # struct name -> (extension name constant, promotion version or None)

    def parse_promotion_version(depends: str) -> Optional[str]:
        """Extract VK_VERSION from depends attribute, e.g., 'VK_VERSION_1_1' -> 'VK_API_VERSION_1_1'."""
        if not depends:
            return None
        # depends can be like "VK_KHR_get_physical_device_properties2,VK_VERSION_1_1"
        # or "(VK_KHR_get_physical_device_properties2,VK_VERSION_1_1)+VK_KHR_shader_float16_int8"
        # We look for VK_VERSION_X_Y patterns
        import re

        match = re.search(r"VK_VERSION_(\d+)_(\d+)", depends)
        if match:
            major, minor = match.groups()
            return f"VK_API_VERSION_{major}_{minor}"
        return None

    # Find extension definitions
    for ext in xml_root.findall("extensions/extension"):
        ext_supported = ext.get("supported", "")

        # Skip extensions not for Vulkan
        if "vulkan" not in ext_supported.split(","):
            continue

        # Skip platform-specific extensions (like Android, Win32, etc.)
        if ext.get("platform") is not None:
            continue

        # Find the EXTENSION_NAME enum in the require block
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

        # Use the promotedto attribute to find promotion version
        promotedto = ext.get("promotedto", "")
        promotion_version = parse_promotion_version(promotedto) if promotedto else None

        for req in ext.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if type_name:
                    extension_map[type_name] = (ext_name_macro, promotion_version)

    # Add aliases to extension_map
    # When VkPhysicalDevice16BitStorageFeaturesKHR is in the map,
    # also add VkPhysicalDevice16BitStorageFeatures (the alias)
    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("category") != "struct":
            continue

        # Check if this is an alias definition
        alias_of = type_elem.get("alias")
        type_name = type_elem.get("name")

        if alias_of and type_name:
            # If the aliased type has extension info, copy it to this type
            if alias_of in extension_map:
                extension_map[type_name] = extension_map[alias_of]
            # Also handle reverse: if this type has info, copy to alias
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

        # Skip types in the skiplist (platform-specific or non-vulkan)
        if vk_name in skiplist:
            continue

        # Skip alias definitions
        if type_elem.get("alias") is not None:
            continue

        # Get the sType from the struct members
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
            # Compute sType from name if not found
            stype = get_stype_from_name(vk_name)

        cpp_name = vk_name_to_cpp_name(vk_name)

        # Get extension info (extension name, promotion version)
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

    # Add VkPhysicalDeviceFeatures2 (its booleans are nested in VkPhysicalDeviceFeatures)
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


def generate_member_name(cpp_name: str) -> str:
    """Generate member variable name from struct name.

    PhysicalDeviceRayTracingPipelineFeaturesKHR -> m_ray_tracing_pipeline_features_khr
    PhysicalDeviceFeatures2 -> m_features_2
    """
    # Remove PhysicalDevice prefix
    name = cpp_name.removeprefix("PhysicalDevice")
    # Convert to snake_case
    snake = to_snake_case(name)
    return f"m_{snake}"


def generate_getter_name(cpp_name: str) -> str:
    """Generate getter method name from struct name.

    PhysicalDeviceRayTracingPipelineFeaturesKHR -> get_ray_tracing_pipeline_features_khr
    PhysicalDeviceFeatures2 -> get_features_2
    """
    # Remove PhysicalDevice prefix
    name = cpp_name.removeprefix("PhysicalDevice")
    # Convert to snake_case
    snake = to_snake_case(name)
    return f"get_{snake}"


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

    # Generate type checks for all features
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

    # Generate alias getters for backwards compatibility
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


def generate_implementation(features: list[FeatureStruct]) -> str:
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

    # Generate chain building for each feature (except Features2)
    for feat in sorted_features:
        member_name = generate_member_name(feat.cpp_name)
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum)

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
        stype_enum = features2.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum)
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
        stype_camel = "e" + to_camel_case(stype_enum)
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
        stype_camel = "e" + to_camel_case(stype_enum)
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
        stype_camel = "e" + to_camel_case(stype_enum)

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

    # Reverse order so Features2 ends up at the head
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

    # Check each feature struct for enabled features
    for feat in sorted(features, key=lambda f: f.cpp_name):
        if not feat.extension:  # Skip features without extensions
            continue

        member_name = generate_member_name(feat.cpp_name)
        prefix = feat.member_prefix

        if feat.members:
            # Check if any feature in this struct is enabled
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
        stype_camel = "e" + to_camel_case(stype_enum)
        lines.append(
            f'        {{"{feat.cpp_name}", vk::StructureType::{stype_camel}}},'
        )

        # Also add short name for convenience
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
    xml_root = parse_vulkan_spec()

    print("Finding feature structures...")
    features = find_feature_structures(xml_root)
    print(f"Found {len(features)} feature structures")

    # Print some examples
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
    impl_content = generate_implementation(features)
    with open(out_path / "vulkan_features.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Generated {len(features)} feature struct members")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
