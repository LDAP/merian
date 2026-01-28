"""Vulkan specification parsing utilities for extensions and dependencies."""

import re
import xml.etree.ElementTree as ET

from .models import Extension, ExtensionDep


def parse_depends(
    depends: str, ext_name_map: dict[str, str]
) -> list[list[ExtensionDep]]:
    """
    Parse the depends attribute into a list of OR'd dependencies,
    where each OR'd item is a list of AND'd dependencies.

    Examples:
    - "VK_KHR_get_physical_device_properties2,VK_VERSION_1_1" -> [[dep1], [ver1_1]]
    - "(VK_KHR_a+VK_KHR_b),VK_VERSION_1_2" -> [[dep_a, dep_b], [ver1_2]]

    Args:
        depends: The depends attribute string from the Vulkan spec
        ext_name_map: Map of extension names to EXTENSION_NAME macros

    Returns:
        List of OR groups, where each group is a list of AND'ed dependencies
    """
    if not depends:
        return []

    result = []

    # Split by comma for OR (but not inside parentheses)
    or_parts = []
    depth = 0
    current = ""
    for c in depends:
        if c == "(":
            depth += 1
            current += c
        elif c == ")":
            depth -= 1
            current += c
        elif c == "," and depth == 0:
            or_parts.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        or_parts.append(current.strip())

    for or_part in or_parts:
        # Remove outer parentheses if present
        or_part = or_part.strip()
        if or_part.startswith("(") and or_part.endswith(")"):
            or_part = or_part[1:-1]

        # Split by '+' for AND
        and_parts = or_part.split("+")
        and_deps = []

        for and_part in and_parts:
            and_part = and_part.strip()
            # Remove any remaining parentheses
            and_part = and_part.strip("()")

            if and_part.startswith("VK_VERSION_"):
                # It's a version requirement
                match = re.match(r"VK_VERSION_(\d+)_(\d+)", and_part)
                if match:
                    major, minor = match.groups()
                    and_deps.append(
                        ExtensionDep(version=f"VK_API_VERSION_{major}_{minor}")
                    )
            elif and_part.startswith("VK_"):
                # It's an extension requirement
                ext_macro = ext_name_map.get(and_part)
                if ext_macro:
                    and_deps.append(ExtensionDep(extension=ext_macro))

        if and_deps:
            result.append(and_deps)

    return result


def build_extension_name_map(xml_root: ET.Element) -> dict[str, str]:
    """
    Build a map of extension name to EXTENSION_NAME macro.

    Args:
        xml_root: Root element of the Vulkan spec XML

    Returns:
        Map of extension names (e.g., VK_KHR_swapchain) to their
        EXTENSION_NAME macro (e.g., VK_KHR_SWAPCHAIN_EXTENSION_NAME)
    """
    ext_name_map = {}

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

    return ext_name_map


def find_extensions(xml_root: ET.Element) -> list[Extension]:
    """
    Find all Vulkan extensions and their dependencies.

    Args:
        xml_root: Root element of the Vulkan spec XML

    Returns:
        List of Extension objects with dependencies parsed
    """
    extensions = []

    # Build the extension name map first
    ext_name_map = build_extension_name_map(xml_root)

    # Now parse extensions with their dependencies
    for ext in xml_root.findall("extensions/extension"):
        ext_name = ext.get("name")
        type = ext.get("type", "")
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

        extensions.append(
            Extension(
                name=ext_name,
                name_macro=ext_name_macro,
                type=type,
                dependencies=dependencies,
            )
        )

    return extensions
