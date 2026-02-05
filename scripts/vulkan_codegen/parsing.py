"""Vulkan specification parsing utilities for extensions and dependencies."""

import re
import xml.etree.ElementTree as ET
from dataclasses import replace

from .codegen import build_extension_name_map, version_to_api_version
from .models import Extension, VulkanStruct, FeatureMember
from .naming import to_camel_case
from .spec import FEATURE_STRUCT_BASE, PROPERTY_STRUCT_BASE


def parse_depends(
    depends: str
) -> list[str]:
    """
    Parse the depends attribute into a list of OR'd dependencies,
    where each OR'd item is a list of AND'd dependencies.

    Examples:
    - "VK_KHR_get_physical_device_properties2,VK_VERSION_1_1" -> [VK_KHR_get_physical_device_properties2]
    - "(VK_KHR_a+VK_KHR_b),VK_VERSION_1_2" -> [VK_KHR_a, VK_KHR_b]

    Args:
        depends: The depends attribute string from the Vulkan spec

    Returns:
        Only the extensions from the arithmetic expression. The ORs can be reconstructed by checking the promoted version.
    """
    if not depends:
        return []

    result = []

    parts = re.split(r"[(),+]", depends)
    for part in parts:
        if part and not part.startswith("VK_VERSION"):
            result.append(part)

    return list(set(result))


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

        # Parse extension number
        ext_number = ext.get("number")
        extension_number = None
        if ext_number:
            try:
                extension_number = int(ext_number)
            except ValueError:
                pass

        # Parse dependencies
        depends = ext.get("depends", "")
        dependencies = parse_depends(depends)

        # Parse promoted version
        promotedto = ext.get("promotedto", "")
        promotedto = version_to_api_version(promotedto)

        # Parse deprecatedby
        deprecatedby = ext.get("deprecatedby", "")
        if deprecatedby.startswith("VK_VERSION"):
            deprecatedby = version_to_api_version(deprecatedby)

        extensions.append(
            Extension(
                name=ext_name,
                name_macro=ext_name_macro,
                type=type,
                extension_number=extension_number,
                dependencies=dependencies,
                promotedto=promotedto,
                deprecatedby=deprecatedby,
            )
        )

    return extensions


def _resolve_type_alias(xml_root: ET.Element, type_name: str) -> ET.Element | None:
    """
    Resolve a type to its actual definition, following aliases if necessary.

    Args:
        xml_root: Root element of the Vulkan spec XML
        type_name: Name of the type to resolve

    Returns:
        The type definition element, or None if not found
    """
    typedef = None
    for t in xml_root.findall("types/type"):
        if t.get("name") == type_name:
            typedef = t
            break

    if typedef is None:
        return None

    # If this is an alias, follow it to the target
    alias_target = typedef.get("alias")
    if alias_target:
        return _resolve_type_alias(xml_root, alias_target)

    return typedef


def enrich_extensions_with_struct_types(
    extensions: list[Extension], xml_root: ET.Element, tags: list[str]
) -> None:
    """
    Enrich Extension objects with property and feature structure types.

    Handles type aliases by resolving them to their target definitions.

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

                # Find the type definition (follow aliases)
                typedef = _resolve_type_alias(xml_root, type_name)
                if not typedef:
                    continue

                struct_extends = typedef.get("structextends", "")

                # Check if it's a property struct
                if PROPERTY_STRUCT_BASE in struct_extends:
                    stype = _extract_stype(typedef, typedef.get("name"), tags)
                    if stype and stype not in ext_obj.property_types:
                        ext_obj.property_types.append(stype)

                # Check if it's a feature struct
                if FEATURE_STRUCT_BASE in struct_extends:
                    stype = _extract_stype(typedef, typedef.get("name"), tags)
                    if stype and stype not in ext_obj.feature_types:
                        ext_obj.feature_types.append(stype)
                break


def build_extension_map(xml_root: ET.Element, tags: list[str]) -> dict[str, Extension]:
    """
    Build a map of extension_name -> Extension object.

    Args:
        xml_root: Root element of the Vulkan spec XML
        tags: List of vendor tags for proper casing

    Returns:
        Dict mapping extension names (e.g., "VK_KHR_swapchain") to Extension objects
    """
    extensions = find_extensions(xml_root)
    enrich_extensions_with_struct_types(extensions, xml_root, tags)
    return {ext.name: ext for ext in extensions}


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


def find_all_structures(xml_root: ET.Element, tags: list[str], skiplist: set[str]) -> list[VulkanStruct]:
    """
    Extract ALL Vulkan struct metadata from XML without filtering by structextends.

    Returns VulkanStruct objects for all struct types including:
    - Feature structs (with VkBool32 members extracted)
    - Property structs (with all members)
    - Base struct VkPhysicalDeviceFeatures2 (special case: members from VkPhysicalDeviceFeatures, adds DeviceCreateInfo to structextends)
    - Alias structs (copies of canonical structs with is_alias=True)

    Args:
        xml_root: Root element of Vulkan XML spec
        tags: Vendor tags for proper naming
        skiplist: Platform-specific types to skip

    Returns:
        List of VulkanStruct with complete metadata including aliases

    Note:
        No manual hacks needed - VkPhysicalDeviceFeatures2 handled in main loop.
    """
    from .codegen import (
        build_extension_type_map,
        build_feature_version_map,
        build_alias_maps,
    )
    from .naming import (
        get_stype_from_name,
        vk_name_to_cpp_name,
    )
    from .spec import DEVICE_CREATE_INFO

    # Build helper maps
    extension_type_map = build_extension_type_map(xml_root)
    version_map = build_feature_version_map(xml_root)
    alias_to_canonical, canonical_to_aliases = build_alias_maps(xml_root)

    structs = []
    canonical_structs = {}  # Map vk_name -> VulkanStruct for later alias copying

    # Extract all struct types from XML (skip aliases in first pass)
    for type_elem in xml_root.findall("types/type"):
        if type_elem.get("category") != "struct":
            continue

        vk_name = type_elem.get("name")
        if not vk_name or vk_name in skiplist or type_elem.get("alias"):
            continue  # Skip aliases in first pass - we'll create them later

        # Extract structextends (critical for later filtering)
        struct_extends = type_elem.get("structextends", "")
        structextends = struct_extends.split(",") if struct_extends else []

        # Special handling for VkPhysicalDeviceFeatures2
        if vk_name == "VkPhysicalDeviceFeatures2":
            # Add DeviceCreateInfo to structextends (not declared in XML)
            structextends.append(DEVICE_CREATE_INFO)

        # Extract sType
        stype = None
        for member in type_elem.findall("member"):
            member_name = member.find("name")
            if member_name is not None and member_name.text == "sType":
                if values := member.get("values"):
                    stype = values
                    break
        if not stype:
            stype = get_stype_from_name(vk_name)

        # Extract ALL members as (type, name, comment) tuples
        members = []
        for member in type_elem.findall("member"):
            member_type = member.find("type")
            member_name = member.find("name")
            if member_type is not None and member_name is not None:
                # Skip sType and pNext
                if member_name.text not in ("sType", "pNext"):
                    comment = member.get("comment", "")
                    members.append((member_type.text, member_name.text, comment))

        # Get extension and version metadata
        ext_info = extension_type_map.get(vk_name)
        extension_name = ext_info[1] if ext_info else None

        vulkan_struct = VulkanStruct(
            vk_name=vk_name,
            cpp_name=vk_name_to_cpp_name(vk_name),
            stype=stype,
            structextends=structextends,
            extension_name=extension_name,
            members=members,  # (type, name, comment) tuples for all members
            aliases=canonical_to_aliases.get(vk_name, []),  # Populate aliases list
            is_alias=False,  # Canonical struct
        )

        structs.append(vulkan_struct)
        canonical_structs[vk_name] = vulkan_struct

    # Second pass: Create alias structs by copying canonical structs
    for alias_name, canonical_name in alias_to_canonical.items():
        if canonical_name not in canonical_structs:
            continue  # Skip if canonical struct wasn't extracted (e.g., filtered by skiplist)

        canonical = canonical_structs[canonical_name]

        # Create copy with alias name
        alias_struct = replace(
            canonical,
            vk_name=alias_name,
            cpp_name=vk_name_to_cpp_name(alias_name),
            aliases=[],  # Aliases don't have their own aliases
            is_alias=True,  # Mark as alias
        )

        structs.append(alias_struct)

    return structs
