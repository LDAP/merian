"""Vulkan specification parsing utilities for extensions and dependencies."""

import re
import xml.etree.ElementTree as ET

from .codegen import build_extension_name_map
from .models import Extension, ExtensionDep
from .naming import to_camel_case


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

        promotedto = ext.get("promotedto", "")
        promoted_version = None
        if promotedto:
            match = re.match(r"VK_VERSION_(\d+)_(\d+)", promotedto)
            if match:
                major, minor = match.groups()
                promoted_version = f"VK_API_VERSION_{major}_{minor}"

        extensions.append(
            Extension(
                name=ext_name,
                name_macro=ext_name_macro,
                type=type,
                dependencies=dependencies,
                promotedto=promoted_version,
            )
        )

    return extensions


def enrich_extensions_with_struct_types(
    extensions: list[Extension], xml_root: ET.Element, tags: list[str]
) -> None:
    """
    Enrich Extension objects with property and feature structure types.

    Args:
        extensions: List of Extension objects to enrich (modified in place)
        xml_root: Root element of the Vulkan spec XML
        tags: List of vendor tags for proper casing
    """
    ext_name_to_ext = {ext.name_macro: ext for ext in extensions}

    for ext_elem in xml_root.findall("extensions/extension"):
        ext_name = ext_elem.get("name")
        ext_supported = ext_elem.get("supported", "")

        if "vulkan" not in ext_supported.split(","):
            continue
        if ext_elem.get("platform") is not None:
            continue

        # Find the extension object
        ext_obj = None
        for req in ext_elem.findall("require"):
            for enum_elem in req.findall("enum"):
                enum_name = enum_elem.get("name", "")
                if enum_name.endswith("_EXTENSION_NAME"):
                    ext_obj = ext_name_to_ext.get(enum_name)
                    break
            if ext_obj:
                break

        if not ext_obj:
            continue

        # Find types required by this extension
        for req in ext_elem.findall("require"):
            for type_elem in req.findall("type"):
                type_name = type_elem.get("name")
                if not type_name:
                    continue

                # Find the type definition
                for typedef in xml_root.findall("types/type"):
                    if typedef.get("name") != type_name:
                        continue

                    struct_extends = typedef.get("structextends", "")

                    # Check if it's a property struct
                    if "VkPhysicalDeviceProperties2" in struct_extends:
                        stype = _extract_stype(typedef, type_name, tags)
                        if stype and stype not in ext_obj.property_types:
                            ext_obj.property_types.append(stype)

                    # Check if it's a feature struct
                    if "VkPhysicalDeviceFeatures2" in struct_extends:
                        stype = _extract_stype(typedef, type_name, tags)
                        if stype and stype not in ext_obj.feature_types:
                            ext_obj.feature_types.append(stype)
                    break


def _extract_stype(typedef: ET.Element, type_name: str, tags: list[str]) -> str | None:
    """Extract sType enum value from a struct definition."""
    for member in typedef.findall("member"):
        member_name_elem = member.find("name")
        if member_name_elem is not None and member_name_elem.text == "sType":
            stype_value = member.get("values")
            if stype_value:
                enum_name = "e" + to_camel_case(
                    stype_value.replace("VK_STRUCTURE_TYPE_", ""), tags
                )
                return enum_name
    return None
