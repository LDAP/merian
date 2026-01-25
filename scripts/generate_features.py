#!/usr/bin/env python3
"""
Generate C++ Feature classes from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. A derived Feature class for each Vulkan feature structure
2. get_feature() function implementations

Similar to generate_enum_string_map.py, this downloads and parses the official
Vulkan specification to generate type-safe C++ code.
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

        # Parse the depends attribute to find promotion version
        depends = ext.get("depends", "")
        promotion_version = parse_promotion_version(depends)

        for req in ext.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if type_name:
                    extension_map[type_name] = (ext_name_macro, promotion_version)

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


def generate_feature_class_name(cpp_name: str) -> str:
    """Generate a Feature class name from the structure name.

    Keeps vendor tags to avoid collisions (e.g., both EXT and NV versions might exist).
    """
    name = cpp_name
    vendor_tag = ""

    # Extract and preserve vendor tag (loaded from spec)
    for tag in tags:
        if name.endswith(f"Features{tag}"):
            vendor_tag = tag
            name = name.removesuffix(f"Features{tag}")
            break
    else:
        # No vendor tag, just remove "Features"
        if name.endswith("Features"):
            name = name.removesuffix("Features")

    if vendor_tag:
        return f"Feature{name}{vendor_tag}"
    return f"Feature{name}"


def generate_header(features: list[FeatureStruct]) -> str:
    """Generate the features.hpp header file content."""
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
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "/**",
        " * @brief Base class for Vulkan feature structures.",
        " *",
        " * Provides a uniform interface for querying and enabling Vulkan features.",
        " * Each derived class wraps a specific VkPhysicalDevice*Features* structure.",
        " */",
        "class Feature {",
        "  public:",
        "    virtual ~Feature() = default;",
        "",
        "    /**",
        "     * @brief Get the list of device extensions required for this feature.",
        "     * @param vk_api_version The Vulkan API version being used.",
        "     * @return Vector of extension name strings (may be empty for core features).",
        "     */",
        "    virtual std::vector<const char*> get_required_extensions(uint32_t vk_api_version) const = 0;",
        "",
        "    /**",
        "     * @brief Get the names of all boolean features in this structure.",
        "     * @return Vector of feature name strings.",
        "     */",
        "    virtual std::vector<const char*> get_feature_names() const = 0;",
        "",
        "    /**",
        "     * @brief Get the value of a specific feature by name.",
        "     * @param name The name of the feature.",
        "     * @return The feature value, or false if not found.",
        "     */",
        "    virtual bool get_feature(const std::string& name) const = 0;",
        "",
        "    /**",
        "     * @brief Set the value of a specific feature by name.",
        "     * @param name The name of the feature.",
        "     * @param value The value to set.",
        "     * @return True if the feature was found and set, false otherwise.",
        "     */",
        "    virtual bool set_feature(const std::string& name, bool value) = 0;",
        "",
        "    /**",
        "     * @brief Get the human-readable name of this feature structure.",
        "     * @return The feature structure name.",
        "     */",
        "    virtual const char* get_name() const = 0;",
        "",
        "    /**",
        "     * @brief Get the Vulkan structure type for this feature.",
        "     * @return The vk::StructureType enum value.",
        "     */",
        "    virtual vk::StructureType get_structure_type() const = 0;",
        "",
        "    /**",
        "     * @brief Get a pointer to the underlying Vulkan structure (for pNext chaining).",
        "     * @return Void pointer to the feature structure.",
        "     */",
        "    virtual void* get_structure_ptr() = 0;",
        "",
        "    /**",
        "     * @brief Set the pNext pointer of the underlying structure.",
        "     * @param p_next The pNext pointer to set.",
        "     */",
        "    virtual void set_pnext(void* p_next) = 0;",
        "};",
        "",
        "using FeatureHandle = std::shared_ptr<Feature>;",
        "",
        "/**",
        " * @brief Get a Feature instance by Vulkan structure type.",
        " * @param stype The vk::StructureType of the feature.",
        " * @return A shared pointer to the Feature, or nullptr if not found.",
        " */",
        "FeatureHandle get_feature(vk::StructureType stype);",
        "",
        "/**",
        " * @brief Get a Feature instance by name.",
        ' * @param name The name of the feature structure (e.g., "PhysicalDeviceRobustness2FeaturesEXT").',
        " * @return A shared pointer to the Feature, or nullptr if not found.",
        " */",
        "FeatureHandle get_feature(const std::string& name);",
        "",
    ]

    # Generate forward declarations for all feature classes
    lines.append("// Forward declarations for feature classes")
    for feat in sorted(features, key=lambda f: f.cpp_name):
        class_name = generate_feature_class_name(feat.cpp_name)
        lines.append(f"class {class_name};")

    lines.extend(
        [
            "",
            "} // namespace merian",
            "",
        ]
    )

    return "\n".join(lines)


def generate_implementation(features: list[FeatureStruct]) -> str:
    """Generate the feature.cpp implementation file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        '#include "merian/vk/utils/features.hpp"',
        "",
        "#include <unordered_map>",
        "#include <functional>",
        "",
        "namespace merian {",
        "",
    ]

    # Generate feature classes
    for feat in sorted(features, key=lambda f: f.cpp_name):
        class_name = generate_feature_class_name(feat.cpp_name)

        lines.append(f"// {'-' * 70}")
        lines.append(f"// {class_name}")
        lines.append(f"// Wraps: vk::{feat.cpp_name}")
        if feat.extension:
            lines.append(f"// Extension: {feat.extension}")
        if feat.promotion_version:
            lines.append(f"// Core in: {feat.promotion_version}")
        lines.append(f"// {'-' * 70}")
        lines.append("")

        lines.append(f"class {class_name} : public Feature {{")
        lines.append("  public:")

        # get_required_extensions
        lines.append(
            "    std::vector<const char*> get_required_extensions(uint32_t vk_api_version) const override {"
        )
        if feat.extension and feat.promotion_version:
            # Feature is promoted to core in some version
            lines.append(f"        if (vk_api_version >= {feat.promotion_version}) {{")
            lines.append("            return {};")
            lines.append("        }")
            lines.append(f"        return {{{feat.extension}}};")
        elif feat.extension:
            lines.append("        (void)vk_api_version;")
            lines.append(f"        return {{{feat.extension}}};")
        else:
            lines.append("        (void)vk_api_version;")
            lines.append("        return {};")
        lines.append("    }")
        lines.append("")

        # get_feature_names
        lines.append(
            "    std::vector<const char*> get_feature_names() const override {"
        )
        lines.append("        return {")
        for member in feat.members:
            lines.append(f'            "{member.name}",')
        lines.append("        };")
        lines.append("    }")
        lines.append("")

        # get_feature
        lines.append("    bool get_feature(const std::string& name) const override {")
        for member in feat.members:
            lines.append(
                f'        if (name == "{member.name}") return features.{member.name} == VK_TRUE;'
            )
        lines.append("        return false;")
        lines.append("    }")
        lines.append("")

        # set_feature
        lines.append(
            "    bool set_feature(const std::string& name, bool value) override {"
        )
        for member in feat.members:
            lines.append(
                f'        if (name == "{member.name}") {{ features.{member.name} = value ? VK_TRUE : VK_FALSE; return true; }}'
            )
        lines.append("        return false;")
        lines.append("    }")
        lines.append("")

        # get_name
        lines.append("    const char* get_name() const override {")
        lines.append(f'        return "{feat.cpp_name}";')
        lines.append("    }")
        lines.append("")

        # get_structure_type
        lines.append("    vk::StructureType get_structure_type() const override {")
        # Convert VK_STRUCTURE_TYPE_* to vk::StructureType::e*
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum)
        lines.append(f"        return vk::StructureType::{stype_camel};")
        lines.append("    }")
        lines.append("")

        # get_structure_ptr
        lines.append("    void* get_structure_ptr() override {")
        lines.append("        return &features;")
        lines.append("    }")
        lines.append("")

        # set_pnext
        lines.append("    void set_pnext(void* p_next) override {")
        lines.append("        features.pNext = p_next;")
        lines.append("    }")
        lines.append("")

        # Accessor for the underlying structure
        lines.append(f"    vk::{feat.cpp_name}& get() {{ return features; }}")
        lines.append(
            f"    const vk::{feat.cpp_name}& get() const {{ return features; }}"
        )
        lines.append("")

        lines.append("  private:")
        lines.append(f"    vk::{feat.cpp_name} features{{}};")
        lines.append("};")
        lines.append("")

    # Generate get_feature(vk::StructureType) function
    lines.append("// " + "-" * 70)
    lines.append("// get_feature implementations")
    lines.append("// " + "-" * 70)
    lines.append("")

    lines.append("FeatureHandle get_feature(vk::StructureType stype) {")
    lines.append("    switch (stype) {")

    for feat in sorted(features, key=lambda f: f.cpp_name):
        class_name = generate_feature_class_name(feat.cpp_name)
        stype_enum = feat.stype.replace("VK_STRUCTURE_TYPE_", "")
        stype_camel = "e" + to_camel_case(stype_enum)
        lines.append(f"        case vk::StructureType::{stype_camel}:")
        lines.append(f"            return std::make_shared<{class_name}>();")

    lines.append("        default:")
    lines.append("            return nullptr;")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Generate get_feature(const std::string&) function
    lines.append("FeatureHandle get_feature(const std::string& name) {")
    lines.append(
        "    static const std::unordered_map<std::string, std::function<FeatureHandle()>> feature_map = {"
    )

    for feat in sorted(features, key=lambda f: f.cpp_name):
        class_name = generate_feature_class_name(feat.cpp_name)
        lines.append(
            f'        {{"{feat.cpp_name}", []() {{ return std::make_shared<{class_name}>(); }}}},'
        )
        # Also add without the "Features" part for convenience
        short_name = feat.cpp_name
        for tag in tags:
            short_name = short_name.removesuffix(tag)
        short_name = short_name.removesuffix("Features")
        short_name = short_name.removeprefix("PhysicalDevice")

        if short_name != feat.cpp_name:
            lines.append(
                f'        {{"{short_name}", []() {{ return std::make_shared<{class_name}>(); }}}},'
            )

    lines.append("    };")
    lines.append("")
    lines.append("    auto it = feature_map.find(name);")
    lines.append("    if (it != feature_map.end()) {")
    lines.append("        return it->second();")
    lines.append("    }")
    lines.append("    return nullptr;")
    lines.append("}")
    lines.append("")

    lines.append("} // namespace merian")
    lines.append("")

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

    print(f"\nGenerating header file: {include_path / 'features.hpp'}")
    header_content = generate_header(features)
    with open(include_path / "features.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'features.cpp'}")
    impl_content = generate_implementation(features)
    with open(out_path / "features.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")


if __name__ == "__main__":
    main()
