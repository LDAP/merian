#!/usr/bin/env python3
"""
Generate C++ extension dependency lookup from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. A function to get extension dependencies for a given extension name

Similar to generate_features.py, this downloads and parses the official
Vulkan specification to generate type-safe C++ code.
"""

import datetime
import re
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


@dataclass
class ExtensionDep:
    """Represents a dependency that can be either an extension or a Vulkan version."""
    extension: Optional[str] = None  # e.g., VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    version: Optional[str] = None  # e.g., VK_API_VERSION_1_1


@dataclass
class Extension:
    """Represents a Vulkan extension."""
    name: str  # e.g., VK_KHR_swapchain
    name_macro: str  # e.g., VK_KHR_SWAPCHAIN_EXTENSION_NAME
    dependencies: list[list[ExtensionDep]] = field(default_factory=list)  # OR of ANDs


def parse_vulkan_spec():
    """Download and parse the Vulkan XML specification."""
    print(f"Downloading Vulkan spec {VULKAN_SPEC_VERSION}...")
    with urlopen(VULKAN_SPEC_URL) as response:
        xml_root = ET.parse(response).getroot()
    return xml_root


def parse_depends(depends: str, ext_name_map: dict[str, str]) -> list[list[ExtensionDep]]:
    """
    Parse the depends attribute into a list of OR'd dependencies,
    where each OR'd item is a list of AND'd dependencies.

    Examples:
    - "VK_KHR_get_physical_device_properties2,VK_VERSION_1_1" -> [[dep1], [ver1_1]]
    - "(VK_KHR_a+VK_KHR_b),VK_VERSION_1_2" -> [[dep_a, dep_b], [ver1_2]]
    """
    if not depends:
        return []

    result = []

    # Split by comma for OR (but not inside parentheses)
    # Simple approach: split by ',' that's not inside parentheses
    or_parts = []
    depth = 0
    current = ""
    for c in depends:
        if c == '(':
            depth += 1
            current += c
        elif c == ')':
            depth -= 1
            current += c
        elif c == ',' and depth == 0:
            or_parts.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        or_parts.append(current.strip())

    for or_part in or_parts:
        # Remove outer parentheses if present
        or_part = or_part.strip()
        if or_part.startswith('(') and or_part.endswith(')'):
            or_part = or_part[1:-1]

        # Split by '+' for AND
        and_parts = or_part.split('+')
        and_deps = []

        for and_part in and_parts:
            and_part = and_part.strip()
            # Remove any remaining parentheses
            and_part = and_part.strip('()')

            if and_part.startswith('VK_VERSION_'):
                # It's a version requirement
                match = re.match(r'VK_VERSION_(\d+)_(\d+)', and_part)
                if match:
                    major, minor = match.groups()
                    and_deps.append(ExtensionDep(version=f"VK_API_VERSION_{major}_{minor}"))
            elif and_part.startswith('VK_'):
                # It's an extension requirement
                ext_macro = ext_name_map.get(and_part)
                if ext_macro:
                    and_deps.append(ExtensionDep(extension=ext_macro))

        if and_deps:
            result.append(and_deps)

    return result


def find_extensions(xml_root) -> list[Extension]:
    """Find all Vulkan extensions and their dependencies."""
    extensions = []

    # Build a map of extension name to EXTENSION_NAME macro
    ext_name_map = {}  # VK_KHR_swapchain -> VK_KHR_SWAPCHAIN_EXTENSION_NAME

    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
        ext_supported = ext.get("supported", "")

        # Skip extensions not for Vulkan
        if "vulkan" not in ext_supported.split(","):
            continue

        # Skip platform-specific extensions
        if ext.get("platform") is not None:
            continue

        # Find the EXTENSION_NAME enum
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

    # Now parse extensions with their dependencies
    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
        ext_supported = ext.get("supported", "")

        # Skip extensions not for Vulkan
        if "vulkan" not in ext_supported.split(","):
            continue

        # Skip platform-specific extensions
        if ext.get("platform") is not None:
            continue

        ext_name_macro = ext_name_map.get(ext_name)
        if not ext_name_macro:
            continue

        depends = ext.get("depends", "")
        dependencies = parse_depends(depends, ext_name_map)

        extensions.append(Extension(
            name=ext_name,
            name_macro=ext_name_macro,
            dependencies=dependencies,
        ))

    return extensions


def generate_header(extensions: list[Extension]) -> str:
    """Generate the extensions.hpp header file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <string>",
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
        "} // namespace merian",
        "",
    ]
    return "\n".join(lines)


def generate_implementation(extensions: list[Extension]) -> str:
    """Generate the extensions.cpp implementation file content."""
    lines = [
        f"// This file was autogenerated for Vulkan {VULKAN_SPEC_VERSION}.",
        f"// Created: {datetime.datetime.now()}",
        "// Do not edit manually!",
        "",
        '#include "merian/vk/utils/extensions.hpp"',
        "",
        "#include <cstring>",
        "#include <vulkan/vulkan.h>",
        "",
        "namespace merian {",
        "",
        "std::vector<const char*> get_extension_dependencies(const char* name, uint32_t vk_api_version) {",
        "    std::vector<const char*> result;",
        "",
    ]

    # Generate if-else chain for each extension
    first = True
    for ext in sorted(extensions, key=lambda e: e.name):
        if not ext.dependencies:
            continue

        condition = "if" if first else "} else if"
        first = False

        lines.append(f"    {condition} (std::strcmp(name, {ext.name_macro}) == 0) {{")

        # Analyze dependencies:
        # - Find version-only OR groups (can satisfy deps without extensions)
        # - Find extension OR groups (need extensions)
        version_only_groups = []
        extension_groups = []

        for and_group in ext.dependencies:
            has_extension = any(d.extension for d in and_group)
            if has_extension:
                extension_groups.append(and_group)
            else:
                # Version-only group
                versions = [d.version for d in and_group if d.version]
                if versions:
                    version_only_groups.append(versions[0])

        if version_only_groups and extension_groups:
            # We have both version alternatives and extension alternatives
            # If any version is satisfied, no extensions needed
            # Sort versions to check highest first (most likely to be satisfied)
            version_only_groups.sort(reverse=True)
            version_check = " && ".join(f"vk_api_version < {v}" for v in version_only_groups)
            lines.append(f"        if ({version_check}) {{")
            # Use the first extension group
            for dep in extension_groups[0]:
                if dep.extension:
                    lines.append(f"            result.push_back({dep.extension});")
            lines.append("        }")
        elif version_only_groups:
            # Only version deps - nothing to add (version satisfies it)
            lines.append("        (void)vk_api_version; // Satisfied by Vulkan version")
        elif extension_groups:
            # Only extension deps - add them
            lines.append("        (void)vk_api_version;")
            for dep in extension_groups[0]:
                if dep.extension:
                    lines.append(f"        result.push_back({dep.extension});")

    if not first:
        lines.append("    }")

    lines.extend([
        "",
        "    return result;",
        "}",
        "",
        "} // namespace merian",
        "",
    ])

    return "\n".join(lines)


def main():
    xml_root = parse_vulkan_spec()

    print("Finding extensions...")
    extensions = find_extensions(xml_root)
    print(f"Found {len(extensions)} extensions")

    # Count extensions with dependencies
    with_deps = [e for e in extensions if e.dependencies]
    print(f"Extensions with dependencies: {len(with_deps)}")

    # Print some examples
    print("\nExample extensions with dependencies:")
    for ext in with_deps[:5]:
        print(f"  - {ext.name}:")
        for or_group in ext.dependencies:
            deps_str = " AND ".join(
                d.extension or d.version or "?" for d in or_group
            )
            print(f"      OR: {deps_str}")

    print(f"\nGenerating header file: {include_path / 'extensions.hpp'}")
    header_content = generate_header(extensions)
    with open(include_path / "extensions.hpp", "w") as f:
        f.write(header_content)

    print(f"Generating implementation file: {out_path / 'extensions.cpp'}")
    impl_content = generate_implementation(extensions)
    with open(out_path / "extensions.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")


if __name__ == "__main__":
    main()
