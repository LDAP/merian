#!/usr/bin/env python3
"""
Generate C++ SPIR-V extensions and capabilities query functions from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. Query functions for available SPIR-V extensions and their requirements
2. Query functions for SPIR-V capabilities and their requirements
3. Support checking for capabilities given device features/properties
4. Feature requirement retrieval for enabling capabilities
"""

import datetime
import re
from dataclasses import dataclass, field
from typing import Optional

from vulkan_codegen.naming import vk_name_to_cpp_name
from vulkan_codegen.spec import (
    VULKAN_SPEC_VERSION,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)


@dataclass
class SpirvExtensionEnable:
    """Represents an enable condition for a SPIR-V extension."""

    version: Optional[str] = None  # e.g., VK_VERSION_1_1
    extension: Optional[str] = None  # e.g., VK_KHR_variable_pointers


@dataclass
class SpirvExtension:
    """Represents a SPIR-V extension."""

    name: str  # e.g., SPV_KHR_variable_pointers
    enables: list[SpirvExtensionEnable] = field(default_factory=list)


@dataclass
class SpirvCapabilityEnable:
    """Represents an enable condition for a SPIR-V capability."""

    # Version-based enable
    version: Optional[str] = None  # e.g., VK_VERSION_1_0

    # Extension-based enable
    extension: Optional[str] = None  # e.g., VK_KHR_variable_pointers

    # Feature-based enable
    feature_struct: Optional[str] = None  # e.g., VkPhysicalDeviceFeatures
    feature_name: Optional[str] = None  # e.g., geometryShader

    # Property-based enable
    property_struct: Optional[str] = None  # e.g., VkPhysicalDeviceVulkan11Properties
    property_member: Optional[str] = None  # e.g., subgroupSupportedOperations
    property_value: Optional[str] = None  # e.g., VK_SUBGROUP_FEATURE_BASIC_BIT

    # Required extension/version for feature/property enables
    requires: Optional[str] = None  # e.g., VK_VERSION_1_0 or VK_KHR_shader_atomic_int64


@dataclass
class SpirvCapability:
    """Represents a SPIR-V capability."""

    name: str  # e.g., Geometry
    enables: list[SpirvCapabilityEnable] = field(default_factory=list)


out_path, include_path = get_output_paths()


def build_extension_name_map(xml_root) -> dict[str, str]:
    """
    Build map of extension name to EXTENSION_NAME macro.

    e.g., VK_EXT_shader_atomic_float2 -> VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME

    Skips provisional/platform-specific extensions as their macros are protected
    by preprocessor defines (e.g., VK_ENABLE_BETA_EXTENSIONS).
    """
    ext_name_map = {}

    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
        ext_supported = ext.get("supported", "")
        if not ext_name:
            continue
        if "vulkan" not in ext_supported.split(","):
            continue
        # Skip platform-specific/provisional extensions
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

        if ext_name_macro:
            ext_name_map[ext_name] = ext_name_macro

    return ext_name_map


def parse_spirv_extensions(xml_root) -> list[SpirvExtension]:
    """Parse <spirvextensions> section from vk.xml."""
    extensions = []

    for ext_elem in xml_root.findall("spirvextensions/spirvextension"):
        name = ext_elem.get("name")
        if not name:
            continue

        ext = SpirvExtension(name=name)

        for enable_elem in ext_elem.findall("enable"):
            enable = SpirvExtensionEnable(
                version=enable_elem.get("version"),
                extension=enable_elem.get("extension"),
            )
            ext.enables.append(enable)

        extensions.append(ext)

    return extensions


def parse_spirv_capabilities(xml_root) -> list[SpirvCapability]:
    """Parse <spirvcapabilities> section from vk.xml."""
    capabilities = []

    for cap_elem in xml_root.findall("spirvcapabilities/spirvcapability"):
        name = cap_elem.get("name")
        if not name:
            continue

        cap = SpirvCapability(name=name)

        for enable_elem in cap_elem.findall("enable"):
            enable = SpirvCapabilityEnable(
                version=enable_elem.get("version"),
                extension=enable_elem.get("extension"),
                feature_struct=enable_elem.get("struct"),
                feature_name=enable_elem.get("feature"),
                property_struct=enable_elem.get("property"),
                property_member=enable_elem.get("member"),
                property_value=enable_elem.get("value"),
                requires=enable_elem.get("requires"),
            )
            cap.enables.append(enable)

        capabilities.append(cap)

    return capabilities


def get_short_feature_struct_name(vk_name: str, tags: list[str]) -> str:
    """
    Convert VkPhysicalDevice*Features* struct name to short form.

    e.g., VkPhysicalDeviceFeatures -> Vulkan10
          VkPhysicalDeviceVulkan12Features -> Vulkan12
          VkPhysicalDeviceRobustness2FeaturesEXT -> Robustness2EXT
    """
    cpp_name = vk_name_to_cpp_name(vk_name)
    short_name = cpp_name

    # Remove vendor tags temporarily for suffix removal
    tag_suffix = ""
    for tag in tags:
        if short_name.endswith(tag):
            tag_suffix = tag
            short_name = short_name[: -len(tag)]
            break

    short_name = short_name.removesuffix("Features")
    short_name = short_name.removeprefix("PhysicalDevice")

    # Special case for core features
    if vk_name == "VkPhysicalDeviceFeatures":
        return "Vulkan10"

    return short_name + tag_suffix


def get_short_property_struct_name(vk_name: str, tags: list[str]) -> str:
    """
    Convert VkPhysicalDevice*Properties* struct name to short form.

    e.g., VkPhysicalDeviceVulkan11Properties -> Vulkan11
          VkPhysicalDeviceSubgroupProperties -> Subgroup
    """
    cpp_name = vk_name_to_cpp_name(vk_name)
    short_name = cpp_name

    # Remove vendor tags temporarily for suffix removal
    tag_suffix = ""
    for tag in tags:
        if short_name.endswith(tag):
            tag_suffix = tag
            short_name = short_name[: -len(tag)]
            break

    short_name = short_name.removesuffix("Properties")
    short_name = short_name.removeprefix("PhysicalDevice")

    return short_name + tag_suffix


def version_to_api_version(version_str: str) -> str:
    """
    Convert VK_VERSION_X_Y to VK_API_VERSION_X_Y.

    e.g., VK_VERSION_1_1 -> VK_API_VERSION_1_1
    """
    return version_str.replace("VK_VERSION_", "VK_API_VERSION_")


def parse_requires(requires_str: str, ext_name_map: dict[str, str]) -> tuple[Optional[str], list[str]]:
    """
    Parse the 'requires' attribute into version and extensions.

    e.g., "VK_VERSION_1_2,VK_KHR_shader_atomic_int64"
          -> (VK_API_VERSION_1_2, [VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME])
    """
    if not requires_str:
        return None, []

    parts = requires_str.split(",")
    version = None
    extensions = []

    for part in parts:
        part = part.strip()
        if part.startswith("VK_VERSION_"):
            version = version_to_api_version(part)
        elif part.startswith("VK_"):
            if part in ext_name_map:
                extensions.append(ext_name_map[part])

    return version, extensions


def generate_header(
    extensions: list[SpirvExtension], capabilities: list[SpirvCapability]
) -> str:
    """Generate the vulkan_spirv.hpp header file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        "#pragma once",
        "",
        '#include "merian/vk/utils/vulkan_features.hpp"',
        '#include "merian/vk/utils/vulkan_properties.hpp"',
        "",
        "#include <cstdint>",
        "#include <string>",
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "/**",
        " * @brief Get list of all SPIR-V extension names.",
        " * @return Vector of SPIR-V extension name strings.",
        " */",
        "std::vector<const char*> get_spirv_extensions();",
        "",
        "/**",
        " * @brief Get Vulkan extension requirements for a SPIR-V extension.",
        " *",
        " * Returns the Vulkan extensions needed to use a SPIR-V extension.",
        " * If the SPIR-V extension is satisfied by the given Vulkan API version,",
        " * returns an empty vector.",
        " *",
        " * @param spirv_extension The SPIR-V extension name (e.g., \"SPV_KHR_variable_pointers\")",
        " * @param vk_api_version The Vulkan API version (e.g., VK_API_VERSION_1_1)",
        " * @return Vector of required Vulkan extension name macros, or empty if satisfied by version",
        " */",
        "std::vector<const char*> get_spirv_extension_requirements(",
        "    const char* spirv_extension,",
        "    uint32_t vk_api_version);",
        "",
        "/**",
        " * @brief Get list of all SPIR-V capability names.",
        " * @return Vector of SPIR-V capability name strings.",
        " */",
        "std::vector<const char*> get_spirv_capabilities();",
        "",
        "/**",
        " * @brief Check if a SPIR-V capability is supported.",
        " *",
        " * Checks if any of the enable conditions for the capability are satisfied.",
        " * This includes checking version requirements, features, and properties.",
        " *",
        " * @param capability The SPIR-V capability name (e.g., \"Geometry\")",
        " * @param vk_api_version The Vulkan API version",
        " * @param features The device features",
        " * @param properties The device properties",
        " * @return true if the capability is supported",
        " */",
        "bool is_spirv_capability_supported(",
        "    const char* capability,",
        "    uint32_t vk_api_version,",
        "    const VulkanFeatures& features,",
        "    const VulkanProperties& properties);",
        "",
        "/**",
        " * @brief Get Vulkan extensions required for a SPIR-V capability.",
        " *",
        " * Returns extensions from enable conditions that are not satisfied by",
        " * the given Vulkan API version.",
        " *",
        " * @param capability The SPIR-V capability name",
        " * @param vk_api_version The Vulkan API version",
        " * @return Vector of required Vulkan extension name macros",
        " */",
        "std::vector<const char*> get_spirv_capability_extensions(",
        "    const char* capability,",
        "    uint32_t vk_api_version);",
        "",
        "/**",
        " * @brief Get features required for a SPIR-V capability.",
        " *",
        ' * Returns feature requirements as "structShortName/featureName" strings',
        " * that can be passed directly to VulkanFeatures::enable_features().",
        " *",
        " * @param capability The SPIR-V capability name",
        " * @param vk_api_version The Vulkan API version",
        " * @return Vector of feature strings in \"structName/featureName\" format",
        " */",
        "std::vector<std::string> get_spirv_capability_features(",
        "    const char* capability,",
        "    uint32_t vk_api_version);",
        "",
        "} // namespace merian",
        "",
    ]

    return "\n".join(lines)


def generate_spirv_extensions_impl(extensions: list[SpirvExtension]) -> list[str]:
    """Generate get_spirv_extensions() implementation."""
    lines = [
        "std::vector<const char*> get_spirv_extensions() {",
        "    return {",
    ]

    for ext in sorted(extensions, key=lambda e: e.name):
        lines.append(f'        "{ext.name}",')

    lines.extend(
        [
            "    };",
            "}",
            "",
        ]
    )

    return lines


def generate_spirv_extension_requirements_impl(
    extensions: list[SpirvExtension],
    ext_name_map: dict[str, str],
) -> list[str]:
    """Generate get_spirv_extension_requirements() implementation."""
    lines = [
        "std::vector<const char*> get_spirv_extension_requirements(",
        "    const char* spirv_extension,",
        "    uint32_t vk_api_version) {",
        "    ",
        "    const std::string_view ext_name{spirv_extension};",
        "    ",
    ]

    first = True
    for ext in sorted(extensions, key=lambda e: e.name):
        keyword = "if" if first else "} else if"
        first = False

        lines.append(f'    {keyword} (ext_name == "{ext.name}") {{')

        # Check if any enable is a version enable
        version_enables = [e for e in ext.enables if e.version]
        extension_enables = [e for e in ext.enables if e.extension and e.extension in ext_name_map]

        if version_enables:
            # If version requirement is met, no extensions needed
            version = version_to_api_version(version_enables[0].version)
            lines.append(f"        if (vk_api_version >= {version}) {{")
            lines.append("            return {};")
            lines.append("        }")

        if extension_enables:
            ext_list = ", ".join(
                ext_name_map[e.extension] for e in extension_enables
            )
            lines.append(f"        return {{{ext_list}}};")
        else:
            lines.append("        return {};")

    lines.extend(
        [
            "    }",
            "    ",
            "    return {};",
            "}",
            "",
        ]
    )

    return lines


def generate_spirv_capabilities_impl(capabilities: list[SpirvCapability]) -> list[str]:
    """Generate get_spirv_capabilities() implementation."""
    lines = [
        "std::vector<const char*> get_spirv_capabilities() {",
        "    return {",
    ]

    for cap in sorted(capabilities, key=lambda c: c.name):
        lines.append(f'        "{cap.name}",')

    lines.extend(
        [
            "    };",
            "}",
            "",
        ]
    )

    return lines


def generate_is_capability_supported_impl(
    capabilities: list[SpirvCapability], tags: list[str]
) -> list[str]:
    """Generate is_spirv_capability_supported() implementation."""
    lines = [
        "bool is_spirv_capability_supported(",
        "    const char* capability,",
        "    uint32_t vk_api_version,",
        "    const VulkanFeatures& features,",
        "    const VulkanProperties& properties) {",
        "    ",
        "    const std::string_view cap_name{capability};",
        "    ",
    ]

    first = True
    for cap in sorted(capabilities, key=lambda c: c.name):
        keyword = "if" if first else "} else if"
        first = False

        lines.append(f'    {keyword} (cap_name == "{cap.name}") {{')

        if not cap.enables:
            lines.append("        return false;")
            continue

        # Generate check for each enable condition (any one being true is sufficient)
        conditions = []

        for enable in cap.enables:
            if enable.version:
                version = version_to_api_version(enable.version)
                conditions.append(f"(vk_api_version >= {version})")

            elif enable.extension:
                # Extension-only enables - caller tracks available extensions
                # We can't check this without an extension list parameter
                pass

            elif enable.feature_struct and enable.feature_name:
                short_name = get_short_feature_struct_name(enable.feature_struct, tags)
                conditions.append(
                    f'features.get_feature("{short_name}", "{enable.feature_name}")'
                )

            elif enable.property_struct and enable.property_member and enable.property_value:
                cpp_struct_name = vk_name_to_cpp_name(enable.property_struct)
                member = enable.property_member
                value = enable.property_value

                # Handle different property value types
                if value == "VK_TRUE":
                    conditions.append(
                        f"(properties.get<vk::{cpp_struct_name}>().{member} == VK_TRUE)"
                    )
                elif value.startswith("VK_"):
                    # It's a flags/enum value - check if the bit is set
                    # Cast to uint32_t to work with vulkan-hpp's vk::Flags wrapper
                    conditions.append(
                        f"(static_cast<uint32_t>(properties.get<vk::{cpp_struct_name}>().{member}) & {value})"
                    )
                else:
                    # Numeric value
                    conditions.append(
                        f"(properties.get<vk::{cpp_struct_name}>().{member} >= {value})"
                    )

        if conditions:
            # Join with || - any condition being true means supported
            condition_str = " ||\n            ".join(conditions)
            lines.append(f"        return {condition_str};")
        else:
            lines.append("        return false;")

    lines.extend(
        [
            "    }",
            "    ",
            "    return false;",
            "}",
            "",
        ]
    )

    return lines


def generate_capability_extensions_impl(
    capabilities: list[SpirvCapability],
    ext_name_map: dict[str, str],
) -> list[str]:
    """Generate get_spirv_capability_extensions() implementation."""
    lines = [
        "std::vector<const char*> get_spirv_capability_extensions(",
        "    const char* capability,",
        "    uint32_t vk_api_version) {",
        "    ",
        "    const std::string_view cap_name{capability};",
        "    std::vector<const char*> result;",
        "    ",
    ]

    first = True
    for cap in sorted(capabilities, key=lambda c: c.name):
        keyword = "if" if first else "} else if"
        first = False

        lines.append(f'    {keyword} (cap_name == "{cap.name}") {{')

        extensions_added = set()

        for enable in cap.enables:
            # Direct extension enables
            if enable.extension and enable.extension in ext_name_map:
                ext_macro = ext_name_map[enable.extension]
                if ext_macro not in extensions_added:
                    extensions_added.add(ext_macro)
                    lines.append(f"        result.push_back({ext_macro});")

            # Extensions from requires attribute
            if enable.requires:
                req_version, req_extensions = parse_requires(enable.requires, ext_name_map)
                for ext_macro in req_extensions:
                    if ext_macro not in extensions_added:
                        extensions_added.add(ext_macro)
                        # Only add if version doesn't satisfy
                        if req_version:
                            lines.append(f"        if (vk_api_version < {req_version}) {{")
                            lines.append(f"            result.push_back({ext_macro});")
                            lines.append("        }")
                        else:
                            lines.append(f"        result.push_back({ext_macro});")

    lines.extend(
        [
            "    }",
            "    ",
            "    return result;",
            "}",
            "",
        ]
    )

    return lines


def generate_capability_features_impl(
    capabilities: list[SpirvCapability], tags: list[str]
) -> list[str]:
    """Generate get_spirv_capability_features() implementation."""
    lines = [
        "std::vector<std::string> get_spirv_capability_features(",
        "    const char* capability,",
        "    [[maybe_unused]] uint32_t vk_api_version) {",
        "    ",
        "    const std::string_view cap_name{capability};",
        "    ",
    ]

    first = True
    for cap in sorted(capabilities, key=lambda c: c.name):
        # Collect feature enables
        feature_enables = [
            e for e in cap.enables if e.feature_struct and e.feature_name
        ]

        if not feature_enables:
            continue

        keyword = "if" if first else "} else if"
        first = False

        lines.append(f'    {keyword} (cap_name == "{cap.name}") {{')
        lines.append("        return {")

        for enable in feature_enables:
            short_name = get_short_feature_struct_name(enable.feature_struct, tags)
            lines.append(f'            "{short_name}/{enable.feature_name}",')

        lines.append("        };")

    if not first:
        lines.append("    }")

    lines.extend(
        [
            "    ",
            "    return {};",
            "}",
            "",
        ]
    )

    return lines


def generate_implementation(
    extensions: list[SpirvExtension],
    capabilities: list[SpirvCapability],
    tags: list[str],
    ext_name_map: dict[str, str],
) -> str:
    """Generate the vulkan_spirv.cpp implementation file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        '#include "merian/vk/utils/vulkan_spirv.hpp"',
        "",
        "#include <string_view>",
        "",
        "namespace merian {",
        "",
    ]

    lines.extend(generate_spirv_extensions_impl(extensions))
    lines.extend(generate_spirv_extension_requirements_impl(extensions, ext_name_map))
    lines.extend(generate_spirv_capabilities_impl(capabilities))
    lines.extend(generate_is_capability_supported_impl(capabilities, tags))
    lines.extend(generate_capability_extensions_impl(capabilities, ext_name_map))
    lines.extend(generate_capability_features_impl(capabilities, tags))

    lines.extend(["} // namespace merian", ""])

    return "\n".join(lines)


def main():
    xml_root = load_vulkan_spec()
    tags = load_vendor_tags(xml_root)

    print("Building extension name map...")
    ext_name_map = build_extension_name_map(xml_root)
    print(f"Found {len(ext_name_map)} Vulkan extension name macros")

    print("Parsing SPIR-V extensions...")
    extensions = parse_spirv_extensions(xml_root)
    print(f"Found {len(extensions)} SPIR-V extensions")

    print("Parsing SPIR-V capabilities...")
    capabilities = parse_spirv_capabilities(xml_root)
    print(f"Found {len(capabilities)} SPIR-V capabilities")

    print("\nExample extensions:")
    for ext in extensions[:3]:
        print(f"  - {ext.name}: {len(ext.enables)} enables")

    print("\nExample capabilities:")
    for cap in capabilities[:5]:
        print(f"  - {cap.name}: {len(cap.enables)} enables")
        for enable in cap.enables[:2]:
            if enable.version:
                print(f"      version: {enable.version}")
            if enable.feature_struct:
                print(f"      feature: {enable.feature_struct}.{enable.feature_name}")
            if enable.property_struct:
                print(
                    f"      property: {enable.property_struct}.{enable.property_member} = {enable.property_value}"
                )

    print(f"\nGenerating header file: {include_path / 'vulkan_spirv.hpp'}")
    header_content = generate_header(extensions, capabilities)
    with open(include_path / "vulkan_spirv.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'vulkan_spirv.cpp'}")
    impl_content = generate_implementation(extensions, capabilities, tags, ext_name_map)
    with open(out_path / "vulkan_spirv.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Generated {len(extensions)} SPIR-V extension entries")
    print(f"Generated {len(capabilities)} SPIR-V capability entries")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
