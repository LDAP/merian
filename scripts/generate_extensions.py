#!/usr/bin/env python3
"""
Generate C++ extension utility functions from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. get_extension_dependencies() - returns dependencies for a given extension
2. get_extension_property_types() - returns property types for an extension
3. get_api_version_property_types() - returns property types for an API version

This is separate from properties/features to keep concerns separated.
"""

import datetime
import re

from vulkan_codegen.parsing import find_extensions
from vulkan_codegen.spec import (
    VULKAN_SPEC_VERSION,
    get_output_paths,
    load_vulkan_spec,
)

out_path, include_path = get_output_paths()


def to_camel_case(name: str) -> str:
    """Convert UPPER_SNAKE_CASE to CamelCase for StructureType enum values."""
    parts = name.split("_")
    result = ""
    for part in parts:
        if part:
            result += part[0].upper() + part[1:].lower()
    return result


def build_extension_name_map(xml_root):
    """Build map of extension name to EXTENSION_NAME macro."""
    ext_name_map = {}

    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
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

        if ext_name and ext_name_macro:
            ext_name_map[ext_name] = ext_name_macro

    return ext_name_map


def find_properties_from_extensions(xml_root, ext_name_map):
    """Find property structures provided by extensions."""
    ext_to_stypes = {}

    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
        ext_supported = ext.get("supported", "")
        if "vulkan" not in ext_supported.split(","):
            continue
        if ext.get("platform") is not None:
            continue

        ext_name_macro = ext_name_map.get(ext_name)
        if not ext_name_macro:
            continue

        for req in ext.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if not type_name:
                    continue

                # Check if this type is a property struct
                for typedef in xml_root.findall("types/type"):
                    if typedef.get("name") == type_name:
                        struct_extends = typedef.get("structextends", "")
                        if "VkPhysicalDeviceProperties2" in struct_extends:
                            # Get the sType
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
                                            )
                                        )
                                        ext_to_stypes.setdefault(
                                            ext_name_macro, []
                                        ).append(enum_name)
                                    break
                        break

    return ext_to_stypes


def find_properties_from_api_versions(xml_root):
    """Find property structures available at each API version."""
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

                # Check if this type is a property struct
                for typedef in xml_root.findall("types/type"):
                    if typedef.get("name") == type_name:
                        struct_extends = typedef.get("structextends", "")
                        if (
                            "VkPhysicalDeviceProperties2" in struct_extends
                            or type_name == "VkPhysicalDeviceProperties2"
                        ):
                            if type_name == "VkPhysicalDeviceProperties2":
                                enum_name = "ePhysicalDeviceProperties2"
                                version_to_stypes.setdefault(api_version, []).append(
                                    enum_name
                                )
                            else:
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
                                                )
                                            )
                                            version_to_stypes.setdefault(
                                                api_version, []
                                            ).append(enum_name)
                                        break
                        break

    return version_to_stypes


def find_property_extension_mapping(
    xml_root,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Find mapping of extensions and API versions to property structure types."""
    ext_name_map = build_extension_name_map(xml_root)
    ext_to_stypes = find_properties_from_extensions(xml_root, ext_name_map)
    version_to_stypes = find_properties_from_api_versions(xml_root)
    return ext_to_stypes, version_to_stypes


def generate_header(extensions) -> str:
    """Generate the vulkan_extensions.hpp header file content."""
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
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "/**",
        " * @brief Get the list of extension dependencies for a given extension.",
        " * @param name The extension name macro (e.g., VK_KHR_SWAPCHAIN_EXTENSION_NAME).",
        " * @param vk_api_version The Vulkan API version being used.",
        " * @return Vector of extension name strings that are required dependencies.",
        " *         Returns empty if no dependencies or if dependencies are satisfied by the API version.",
        " */",
        "std::vector<const char*> get_extension_dependencies(const char* name, uint32_t vk_api_version);",
        "",
        "/**",
        " * @brief Get the property structure types provided by an extension.",
        " * @param name The extension name macro.",
        " * @return Vector of vk::StructureType values for property structs from this extension.",
        " */",
        "std::vector<vk::StructureType> get_extension_property_types(const char* name);",
        "",
        "/**",
        " * @brief Get all property structure types available at a given API version.",
        " * @param vk_api_version The Vulkan API version.",
        " * @return Vector of vk::StructureType values for core property structs (cumulative).",
        " */",
        "std::vector<vk::StructureType> get_api_version_property_types(uint32_t vk_api_version);",
        "",
        "} // namespace merian",
        "",
    ]
    return "\n".join(lines)


def generate_extension_dependencies_impl(extensions) -> list[str]:
    """Generate get_extension_dependencies() implementation."""
    lines = [
        "std::vector<const char*> get_extension_dependencies(const char* name, uint32_t vk_api_version) {",
        "    std::vector<const char*> result;",
        "",
    ]

    first = True
    for ext in sorted(extensions, key=lambda e: e.name):
        if not ext.dependencies:
            continue

        condition = "if" if first else "} else if"
        first = False

        lines.append(f"    {condition} (std::strcmp(name, {ext.name_macro}) == 0) {{")

        # Analyze dependencies
        version_only_groups = []
        extension_groups = []

        for and_group in ext.dependencies:
            has_extension = any(d.extension for d in and_group)
            if has_extension:
                extension_groups.append(and_group)
            else:
                versions = [d.version for d in and_group if d.version]
                if versions:
                    version_only_groups.append(versions[0])

        if version_only_groups and extension_groups:
            version_only_groups.sort(reverse=True)
            version_check = " && ".join(
                f"vk_api_version < {v}" for v in version_only_groups
            )
            lines.append(f"        if ({version_check}) {{")
            for dep in extension_groups[0]:
                if dep.extension:
                    lines.append(f"            result.push_back({dep.extension});")
            lines.append("        }")
        elif version_only_groups:
            lines.append("        (void)vk_api_version; // Satisfied by Vulkan version")
        elif extension_groups:
            lines.append("        (void)vk_api_version;")
            for dep in extension_groups[0]:
                if dep.extension:
                    lines.append(f"        result.push_back({dep.extension});")

    if not first:
        lines.append("    }")

    lines.extend(["", "    return result;", "}", ""])
    return lines


def generate_extension_property_types_impl(ext_to_stypes) -> list[str]:
    """Generate get_extension_property_types() implementation."""
    lines = [
        "std::vector<vk::StructureType> get_extension_property_types(const char* name) {",
        "    std::vector<vk::StructureType> result;",
        "",
    ]

    first = True
    for ext_macro in sorted(ext_to_stypes.keys()):
        stypes = ext_to_stypes[ext_macro]
        condition = "if" if first else "} else if"
        first = False
        lines.append(f"    {condition} (std::strcmp(name, {ext_macro}) == 0) {{")
        for stype_enum in sorted(stypes):
            lines.append(f"        result.push_back(vk::StructureType::{stype_enum});")

    if not first:
        lines.append("    }")

    lines.extend(["", "    return result;", "}", ""])
    return lines


def generate_api_version_property_types_impl(version_to_stypes) -> list[str]:
    """Generate get_api_version_property_types() implementation."""
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


def generate_implementation(
    extensions,
    ext_to_stypes: dict[str, list[str]],
    version_to_stypes: dict[str, list[str]],
) -> str:
    """Generate the vulkan_extensions.cpp implementation file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        '#include "merian/vk/utils/vulkan_extensions.hpp"',
        "",
        "#include <cstring>",
        "#include <vulkan/vulkan.h>",
        "",
        "namespace merian {",
        "",
    ]

    lines.extend(generate_extension_dependencies_impl(extensions))
    lines.extend(generate_extension_property_types_impl(ext_to_stypes))
    lines.extend(generate_api_version_property_types_impl(version_to_stypes))

    lines.extend(["} // namespace merian", ""])

    return "\n".join(lines)


def main():
    xml_root = load_vulkan_spec()

    print("Finding extensions...")
    extensions = find_extensions(xml_root)
    print(f"Found {len(extensions)} extensions")

    with_deps = [e for e in extensions if e.dependencies]
    print(f"Extensions with dependencies: {len(with_deps)}")

    print("Finding property extension mappings...")
    ext_to_stypes, version_to_stypes = find_property_extension_mapping(xml_root)
    print(f"Extensions with properties: {len(ext_to_stypes)}")
    print(f"API versions with properties: {len(version_to_stypes)}")

    print(f"\nGenerating header file: {include_path / 'vulkan_extensions.hpp'}")
    header_content = generate_header(extensions)
    with open(include_path / "vulkan_extensions.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'vulkan_extensions.cpp'}")
    impl_content = generate_implementation(extensions, ext_to_stypes, version_to_stypes)
    with open(out_path / "vulkan_extensions.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
