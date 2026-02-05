#!/usr/bin/env python3
"""
Generate C++ extension info system from the Vulkan specification.

Generates:
- ExtensionType enum (Instance, Device)
- DependencyInfo struct for representing dependencies
- ExtensionInfo struct with name, type, dependencies, promoted version, and property types
- Static storage of all extension info
- get_extension_info() lookup function
- get_api_version_property_types() helper function
"""

from warnings import deprecated
from vulkan_codegen.codegen import generate_file_header
from vulkan_codegen.parsing import enrich_extensions_with_struct_types, find_extensions
from vulkan_codegen.spec import (
    VULKAN_SPEC_VERSION,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)

out_path, include_path = get_output_paths()


def generate_header(extensions) -> str:
    """Generate the vulkan_extensions.hpp header file."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
        "#pragma once",
        "",
        '#include "vulkan/vulkan.hpp"',
        "",
        "#include <cstdint>",
        "#include <span>",
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "enum class ExtensionType : uint8_t {",
        "    Instance,",
        "    Device,",
        "};",
        "",
        "struct ExtensionInfo;",
        "",
        "struct ExtensionInfo {",
        "    const char* name;",
        "    ExtensionType type;",
        "    std::span<const ExtensionInfo* const> dependencies;",
        "    uint32_t promoted_to_version;",
        "    const ExtensionInfo* const deprecated_by;",
        "    std::span<const vk::StructureType> property_types;",
        "    std::span<const vk::StructureType> feature_types;",
        "",
        "    bool is_device_extension() const { return type == ExtensionType::Device; }",
        "    bool is_instance_extension() const { return type == ExtensionType::Instance; }",
        "};",
        "",
        "const ExtensionInfo* get_extension_info(const char* name);",
        "",
    ]

    # Generate forward declarations for all extensions
    lines.append("// Forward declarations for all extensions")
    for ext in sorted(extensions, key=lambda e: e.name):
        var_name = make_var_name(ext.name)
        lines.append(f"extern const ExtensionInfo {var_name};")

    lines.extend(["", "} // namespace merian", ""])
    return "\n".join(lines)


def make_var_name(ext_name: str) -> str:
    """Convert extension name to our extension info C++ variable name."""
    return f"{ext_name.upper()}_INFO"


def generate_static_extension_infos(extensions) -> list[str]:
    """Generate static ExtensionInfo objects."""
    lines = []

    # Build map of extension macro to variable name
    ext_to_var = {
        ext.name: make_var_name(ext.name) for ext in extensions
    }

    # Generate dependency pointer arrays
    for ext in sorted(extensions, key=lambda e: e.name):
        if not ext.dependencies:
            continue

        var_name = make_var_name(ext.name)

        lines.append(f"const ExtensionInfo* const {var_name}_DEPS[] = {{")
        for dep_ext in sorted(ext.dependencies):
            dep_var = ext_to_var.get(dep_ext)
            if dep_var:
                lines.append(f"    &{dep_var},")
            else:
                print(f"WARN: Unknown extension {dep_ext}")
        lines.append("};")

    lines.append("")

    # Generate property type arrays
    for ext in sorted(extensions, key=lambda e: e.name):
        if not ext.property_types:
            continue

        var_name = make_var_name(ext.name)
        lines.append(f"const vk::StructureType {var_name}_PROPERTIES[] = {{")
        for stype in sorted(ext.property_types):
            lines.append(f"    vk::StructureType::{stype},")
        lines.append("};")

    lines.append("")

    # Generate feature type arrays
    for ext in sorted(extensions, key=lambda e: e.name):
        if not ext.feature_types:
            continue

        var_name = make_var_name(ext.name)
        lines.append(f"const vk::StructureType {var_name}_FEATURES[] = {{")
        for stype in sorted(ext.feature_types):
            lines.append(f"    vk::StructureType::{stype},")
        lines.append("};")

    lines.append("")

    # Generate ExtensionInfo objects
    for ext in sorted(extensions, key=lambda e: e.name):
        var_name = make_var_name(ext.name)
        ext_type = (
            "ExtensionType::Device"
            if ext.type == "device"
            else "ExtensionType::Instance"
        )
        promoted = ext.promotedto if ext.promotedto else (ext.deprecatedby if ext.deprecatedby and not ext.deprecatedby.startswith("VK_API_VERSION") else "(uint32_t)-1")
        deprecated_by = f"&{make_var_name(ext.deprecatedby)}" if ext.deprecatedby and not ext.deprecatedby.startswith("VK_API_VERSION") else "{}"

        lines.append(f"const ExtensionInfo {var_name} = {{")
        lines.append(f"    {ext.name_macro},")
        lines.append(f"    {ext_type},")

        if ext.dependencies:
            lines.append(
                f"    std::span<const ExtensionInfo* const>({var_name}_DEPS, {len(ext.dependencies)}),"
            )
        else:
            lines.append("    {},")

        lines.append(f"    {promoted},")
        lines.append(f"    {deprecated_by},")

        if ext.property_types:
            lines.append(
                f"    std::span<const vk::StructureType>({var_name}_PROPERTIES, {len(ext.property_types)}),"
            )
        else:
            lines.append("    {},")

        if ext.feature_types:
            lines.append(
                f"    std::span<const vk::StructureType>({var_name}_FEATURES, {len(ext.feature_types)}),"
            )
        else:
            lines.append("    {},")

        lines.append("};")

    lines.append("")
    return lines


def generate_get_extension_info_impl(extensions) -> list[str]:
    """Generate get_extension_info() implementation."""
    lines = [
        "const ExtensionInfo* get_extension_info(const char* name) {",
    ]

    for ext in sorted(extensions, key=lambda e: e.name):
        var_name = make_var_name(ext.name)
        lines.append(f"    if (std::strcmp(name, {ext.name_macro}) == 0) {{")
        lines.append(f"        return &{var_name};")
        lines.append("    }")

    lines.extend(["    return nullptr;", "}", ""])
    return lines


def generate_implementation(extensions) -> str:
    """Generate the vulkan_extensions.cpp implementation file."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
        '#include "merian/vk/utils/vulkan_extensions.hpp"',
        "",
        "#include <cstring>",
        "#include <vulkan/vulkan.h>",
        "",
        "namespace merian {",
        "",
    ]

    lines.extend(generate_static_extension_infos(extensions))
    lines.extend(generate_get_extension_info_impl(extensions))

    lines.extend(["} // namespace merian", ""])

    return "\n".join(lines)


def main():
    xml_root = load_vulkan_spec()
    tags = load_vendor_tags(xml_root)

    print("Finding extensions...")
    extensions = find_extensions(xml_root)
    print(f"Found {len(extensions)} extensions")

    print("Enriching extensions with property/feature types...")
    enrich_extensions_with_struct_types(extensions, xml_root, tags)

    device_exts = [e for e in extensions if e.type == "device"]
    instance_exts = [e for e in extensions if e.type == "instance"]
    with_deps = [e for e in extensions if e.dependencies]
    promoted = [e for e in extensions if e.promotedto]
    with_props = [e for e in extensions if e.property_types]
    with_features = [e for e in extensions if e.feature_types]

    print(f"  Device extensions: {len(device_exts)}")
    print(f"  Instance extensions: {len(instance_exts)}")
    print(f"  With dependencies: {len(with_deps)}")
    print(f"  Promoted to core: {len(promoted)}")
    print(f"  With property types: {len(with_props)}")
    print(f"  With feature types: {len(with_features)}")

    print(f"\nGenerating header file: {include_path / 'vulkan_extensions.hpp'}")
    header_content = generate_header(extensions)
    with open(include_path / "vulkan_extensions.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'vulkan_extensions.cpp'}")
    impl_content = generate_implementation(extensions)
    with open(out_path / "vulkan_extensions.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
