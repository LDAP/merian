#!/usr/bin/env python3
"""
Generate C++ SPIR-V extensions and capabilities query functions from the Vulkan specification.

This script parses the Vulkan XML specification and generates:
1. Query functions for available SPIR-V extensions and their requirements
2. Query functions for SPIR-V capabilities and their requirements
3. Support checking for capabilities given device features/properties
4. Feature requirement retrieval for enabling capabilities
"""

from dataclasses import dataclass, field
from typing import Optional

from vulkan_codegen.codegen import (
    build_extension_name_map,
    build_extension_type_map,
    build_struct_aggregation_map,
    generate_file_header,
    version_to_api_version,
)
from vulkan_codegen.naming import vk_name_to_cpp_name
from vulkan_codegen.parsing import find_all_structures
from vulkan_codegen.spec import (
    PROPERTY_STRUCT_BASE,
    VULKAN_SPEC_VERSION,
    build_skiplist,
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


def parse_requires(
    requires_str: str, ext_name_map: dict[str, str]
) -> tuple[Optional[str], list[str]]:
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


def build_property_struct_map(xml_root, tags: list[str]) -> dict[str, dict]:
    """
    Build a map of property struct names to their metadata.

    For VulkanXXProperties structs, this finds the corresponding extension structs
    that were promoted/aggregated into them. Extension structs are preferred because
    they're available both when the extension is enabled AND when the Vulkan version
    is high enough.

    Returns:
        dict mapping struct_name -> {
            'vk_name': str,
            'cpp_name': str,
            'members': list[(type, name, comment)],
            'extension_name': Optional[str],
            'core_version': Optional[str],
            'aggregated_by': Optional[str],  # VulkanXX struct that aggregates this
            'aggregates': list[str],  # List of structs that are aggregated into this VulkanXX
        }
    """
    skiplist = build_skiplist(xml_root)
    all_structs = find_all_structures(xml_root, tags, skiplist)

    # Filter for property structs only
    property_structs = [
        s for s in all_structs
        if PROPERTY_STRUCT_BASE in s.structextends
    ]

    # Build the reverse aggregation map: VulkanXX -> list of individual structs
    aggregation_map = build_struct_aggregation_map(xml_root, "Properties")

    # Build reverse map: VulkanXX -> list of structs aggregated into it
    reverse_agg_map = {}
    for struct_name, vulkan_xx in aggregation_map.items():
        reverse_agg_map.setdefault(vulkan_xx, []).append(struct_name)

    # Build extension type map for getting extension info
    extension_type_map = build_extension_type_map(xml_root)

    # Build feature version map to know when structs were added to core
    from vulkan_codegen.codegen import build_feature_version_map
    version_map = build_feature_version_map(xml_root)

    # Build the result map
    struct_map = {}

    for struct in property_structs:
        # Get extension and version info
        ext_info = extension_type_map.get(struct.vk_name)
        extension_name = ext_info[1] if ext_info else struct.extension_name

        # Check when this struct was added to core
        core_version = version_map.get(struct.vk_name)

        # Check if this struct was promoted (has a VulkanXX aggregate)
        aggregated_by = aggregation_map.get(struct.vk_name)

        # Check what structs this aggregates (for VulkanXX structs)
        aggregates = reverse_agg_map.get(struct.vk_name, [])

        struct_map[struct.vk_name] = {
            'vk_name': struct.vk_name,
            'cpp_name': struct.cpp_name,
            'members': struct.members,
            'extension_name': extension_name,
            'core_version': core_version,
            'aggregated_by': aggregated_by,
            'aggregates': aggregates,
        }

    return struct_map


def build_vulkan_property_member_map(
    struct_map: dict[str, dict],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """
    Build a map from (VulkanXXProperties, member_name) to list of (alternative_struct, alternative_member).

    VulkanXXProperties structs aggregate members from multiple property structs, often adding
    prefixes to avoid naming conflicts. For example:
    - VkPhysicalDeviceVulkan11Properties.subgroupSupportedOperations
    - VkPhysicalDeviceSubgroupProperties.supportedOperations (same property, different name!)

    This function matches members by type to find these mappings, since the aggregated structs
    don't preserve the exact member names.

    Returns:
        dict mapping (vulkan_struct_name, member_name) -> [(alt_struct, alt_member), ...]
    """
    member_map = {}

    # For each VulkanXXProperties struct
    for struct_name, struct_info in struct_map.items():
        if not struct_name.startswith('VkPhysicalDeviceVulkan'):
            continue
        if 'Properties' not in struct_name:
            continue

        vulkan_members = struct_info.get('members', [])

        # For each member in the VulkanXX struct, try to find corresponding members
        # in structs that match by type
        for v_type, v_member, v_comment in vulkan_members:
            alternatives = []

            # Search all other property structs for members with the same type
            for other_name, other_info in struct_map.items():
                if other_name == struct_name:
                    continue
                if other_name.startswith('VkPhysicalDeviceVulkan'):
                    continue  # Skip other VulkanXX structs

                # Check if this struct has a member with matching type
                for o_type, o_member, o_comment in other_info.get('members', []):
                    if o_type == v_type:
                        # Types match - check if names are related
                        # Common patterns: subgroupX -> X, vulkanXX -> XX, etc.
                        if (v_member.lower().endswith(o_member.lower()) or
                            o_member.lower().endswith(v_member.lower()) or
                            v_member.lower().replace('subgroup', '') == o_member.lower() or
                            v_member.lower().replace('vulkan', '') == o_member.lower()):
                            # Likely the same property!
                            core_version = other_info.get('core_version')
                            extension = other_info.get('extension_name')
                            # Prefer earlier versions and extension structs
                            priority = 0
                            if core_version:
                                # Earlier version = higher priority
                                if 'API_VERSION_1_1' in str(core_version):
                                    priority = 3
                                elif 'API_VERSION_1_2' in str(core_version):
                                    priority = 2
                                elif 'API_VERSION_1_3' in str(core_version):
                                    priority = 1
                            if extension:
                                priority += 10  # Extensions get even higher priority
                            alternatives.append((priority, other_name, o_member))

            # Sort by priority (higher first) and store
            if alternatives:
                alternatives.sort(key=lambda x: x[0], reverse=True)
                member_map[(struct_name, v_member)] = [(name, mem) for _, name, mem in alternatives]

    return member_map


def generate_header(
    extensions: list[SpirvExtension], capabilities: list[SpirvCapability]
) -> str:
    """Generate the vulkan_spirv.hpp header file content."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
        "#pragma once",
        "",
        '#include "merian/vk/utils/vulkan_features.hpp"',
        '#include "merian/vk/utils/vulkan_properties.hpp"',
        "",
        "#include <cstdint>",
        "#include <string>",
        "#include <unordered_set>",
        "#include <vector>",
        "",
        "namespace merian {",
        "",
        "/**",
        " * @brief Get list of all SPIR-V extension names.",
        " * @return Vector of SPIR-V extension name strings.",
        " */",
        "const std::vector<const char*>& get_spirv_extensions();",
        "",
        "/**",
        " * @brief Get Vulkan extension requirements for a SPIR-V extension.",
        " *",
        " * Returns the Vulkan extensions needed to use a SPIR-V extension.",
        " * If the SPIR-V extension is satisfied by the given Vulkan API version,",
        " * returns an empty vector.",
        " *",
        ' * @param spirv_extension The SPIR-V extension name (e.g., "SPV_KHR_variable_pointers")',
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
        "const std::vector<const char*>& get_spirv_capabilities();",
        "",
        "/**",
        " * @brief Check if a SPIR-V capability is supported.",
        " *",
        " * Checks if any of the enable conditions for the capability are satisfied.",
        " * This includes checking version requirements, features, and properties.",
        " *",
        ' * @param capability The SPIR-V capability name (e.g., "Geometry")',
        " * @param vk_api_version The Vulkan API version",
        " * @param enabled_extensions Set of enabled Vulkan extension names",
        " * @param features The device features",
        " * @param properties The device properties",
        " * @return true if the capability is supported",
        " */",
        "bool is_spirv_capability_supported(",
        "    const char* capability,",
        "    uint32_t vk_api_version,",
        "    const std::unordered_set<std::string>& enabled_extensions,",
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
        ' * Returns feature requirement names (e.g., "rayTracingPipeline")',
        " * that can be passed directly to VulkanFeatures::enable_features().",
        " *",
        " * @param capability The SPIR-V capability name",
        " * @param vk_api_version The Vulkan API version",
        " * @return Vector of feature name strings",
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
        "const std::vector<const char*>& get_spirv_extensions() {",
        "    static const std::vector<const char*> extensions = {",
    ]

    for ext in sorted(extensions, key=lambda e: e.name):
        lines.append(f'        "{ext.name}",')

    lines.extend(
        [
            "    };",
            "    return extensions;",
            "}",
            "",
        ]
    )

    return lines


def generate_spirv_extension_requirements_impl(
    extensions: list[SpirvExtension],
    ext_name_map: dict[str, str],
) -> list[str]:
    """Generate get_spirv_extension_requirements() implementation using map lookup."""
    lines = [
        "namespace {",
        "struct SpirvExtReq {",
        "    uint32_t satisfied_by_version;",
        "    std::vector<const char*> extensions;",
        "};",
        "",
        "// NOLINTNEXTLINE(cert-err58-cpp)",
        "const std::unordered_map<std::string_view, SpirvExtReq> spirv_ext_req_map = {",
    ]

    for ext in sorted(extensions, key=lambda e: e.name):
        version_enables = [e for e in ext.enables if e.version]
        extension_enables = [
            e for e in ext.enables if e.extension and e.extension in ext_name_map
        ]

        version_str = (
            version_to_api_version(version_enables[0].version)
            if version_enables
            else "0"
        )
        if extension_enables:
            ext_list = ", ".join(ext_name_map[e.extension] for e in extension_enables)
        else:
            ext_list = ""

        lines.append(f'    {{"{ext.name}", {{{version_str}, {{{ext_list}}}}}}},')

    lines.extend(
        [
            "};",
            "} // namespace",
            "",
            "std::vector<const char*> get_spirv_extension_requirements(",
            "    const char* spirv_extension,",
            "    uint32_t vk_api_version) {",
            "",
            "    const auto it = spirv_ext_req_map.find(spirv_extension);",
            "    if (it == spirv_ext_req_map.end()) {",
            "        return {};",
            "    }",
            "    if (it->second.satisfied_by_version != 0 &&",
            "        vk_api_version >= it->second.satisfied_by_version) {",
            "        return {};",
            "    }",
            "    return it->second.extensions;",
            "}",
            "",
        ]
    )

    return lines


def generate_spirv_capabilities_impl(capabilities: list[SpirvCapability]) -> list[str]:
    """Generate get_spirv_capabilities() implementation."""
    lines = [
        "const std::vector<const char*>& get_spirv_capabilities() {",
        "    static const std::vector<const char*> capabilities = {",
    ]

    for cap in sorted(capabilities, key=lambda c: c.name):
        lines.append(f'        "{cap.name}",')

    lines.extend(
        [
            "    };",
            "    return capabilities;",
            "}",
            "",
        ]
    )

    return lines


def sanitize_capability_name(cap_name: str) -> str:
    """Convert SPIR-V capability name to valid C++ function name."""
    # Most capability names are already valid C++ identifiers
    # Just need to handle edge cases like special characters
    sanitized = cap_name.replace('.', '_').replace('-', '_')
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = 'Cap' + sanitized
    return sanitized


def has_checkable_enables(cap: SpirvCapability) -> bool:
    """Check if capability has any enables that can be checked at runtime."""
    for enable in cap.enables:
        # Version/extension/feature/property enables can be checked
        if (enable.version or enable.extension or
            enable.feature_struct or enable.property_struct):
            return True
    return False


def generate_requires_condition(
    requires_str: Optional[str],
    ext_name_map: dict[str, str],
) -> Optional[str]:
    """Generate (version OR ext1 OR ext2) condition from requires field."""
    if not requires_str:
        return None

    req_version, req_extensions = parse_requires(requires_str, ext_name_map)

    conditions = []

    # Add version check
    if req_version:
        conditions.append(f"vk_api_version >= {req_version}")

    # Add extension checks using the extension name macros
    for ext_macro in req_extensions:
        conditions.append(f'enabled_extensions.contains({ext_macro})')

    if not conditions:
        return None

    return " || ".join(conditions)


def generate_grouped_enable_check(
    enables: list[SpirvCapabilityEnable],
    member_map: dict,
    ext_name_map: dict[str, str],
) -> list[str]:
    """Generate combined if-statement for multiple enables with the same requires."""
    lines = []

    # All enables in this group have the same requires
    first_enable = enables[0]
    requires_condition = generate_requires_condition(first_enable.requires, ext_name_map)

    # Start the requires if-block (if any)
    indent = "    "
    if requires_condition:
        lines.append(f"{indent}// Enable: requires ({first_enable.requires})")
        lines.append(f"{indent}if ({requires_condition}) {{")
        indent = "        "
    else:
        lines.append(f"{indent}// Enable: no requires")

    # Generate checks for each enable and OR them together
    any_checks = []
    for enable in enables:
        if enable.version and not enable.feature_struct and not enable.property_struct:
            # Direct version enable
            version = version_to_api_version(enable.version)
            any_checks.append(f"vk_api_version >= {version}")

        elif enable.extension and not enable.feature_struct and not enable.property_struct:
            # Direct extension enable
            ext_macro = ext_name_map.get(enable.extension, f'"{enable.extension}"')
            any_checks.append(f"enabled_extensions.contains({ext_macro})")

        elif enable.feature_struct and enable.feature_name:
            # Feature-based enable
            any_checks.append(f'features.get_feature("{enable.feature_name}")')

        elif enable.property_struct and enable.property_member and enable.property_value:
            # Property-based enable - need to generate inline
            # For properties, we need to check availability first, so we handle this differently
            member = enable.property_member
            value = enable.property_value

            # Get alternative property structs
            structs_to_check = []
            alternatives = member_map.get((enable.property_struct, member), [])
            if alternatives:
                structs_to_check.extend(alternatives)
            structs_to_check.append((enable.property_struct, member))

            # Generate check for each alternative struct
            for struct_name, member_name in structs_to_check:
                cpp_struct_name = vk_name_to_cpp_name(struct_name)

                # Generate value check
                if value == "VK_TRUE":
                    prop_check = f"(properties.is_available<vk::{cpp_struct_name}>() && properties.get<vk::{cpp_struct_name}>().{member_name} == VK_TRUE)"
                elif value.startswith("VK_"):
                    # Bit flag check
                    prop_check = f"(properties.is_available<vk::{cpp_struct_name}>() && (static_cast<uint32_t>(properties.get<vk::{cpp_struct_name}>().{member_name}) & {value}))"
                else:
                    # Numeric comparison
                    prop_check = f"(properties.is_available<vk::{cpp_struct_name}>() && properties.get<vk::{cpp_struct_name}>().{member_name} >= {value})"

                any_checks.append(prop_check)

    # Combine all checks with OR
    if any_checks:
        if len(any_checks) == 1:
            lines.append(f"{indent}if ({any_checks[0]}) {{")
        else:
            # Format nicely with line breaks for readability
            lines.append(f"{indent}if ({any_checks[0]} ||")
            for check in any_checks[1:-1]:
                lines.append(f"{indent}    {check} ||")
            lines.append(f"{indent}    {any_checks[-1]}) {{")
        lines.append(f"{indent}    return true;")
        lines.append(f"{indent}}}")

    # Close requires if-block
    if requires_condition:
        lines.append("    }")

    lines.append("")  # Blank line between enables
    return lines


def generate_is_capability_supported_impl(
    capabilities: list[SpirvCapability],
    tags: list[str],
    struct_map: dict[str, dict],
    ext_name_map: dict[str, str],
) -> list[str]:
    """Generate is_spirv_capability_supported() implementation with readable if-statements."""
    lines = [
        "namespace {",
        "",
    ]

    # Build member mapping for VulkanXX properties
    member_map = build_vulkan_property_member_map(struct_map)

    # Generate a check function for each capability
    map_entries = []

    for cap in sorted(capabilities, key=lambda c: c.name):
        # Skip capabilities with no checkable enables
        if not has_checkable_enables(cap):
            continue

        # Sanitize capability name for C++ function
        func_name = f"check_capability_{sanitize_capability_name(cap.name)}"
        map_entries.append((cap.name, func_name))

        # Generate function header
        lines.extend([
            f"// {cap.name}",
            f"bool {func_name}(",
            "    [[maybe_unused]] uint32_t vk_api_version,",
            "    [[maybe_unused]] const std::unordered_set<std::string>& enabled_extensions,",
            "    [[maybe_unused]] const VulkanFeatures& features,",
            "    [[maybe_unused]] const VulkanProperties& properties) {",
            "",
        ])

        # Group enables by their requires field to avoid redundant checks
        enables_by_requires = {}
        for enable in cap.enables:
            req_key = enable.requires or ""
            if req_key not in enables_by_requires:
                enables_by_requires[req_key] = []
            enables_by_requires[req_key].append(enable)

        # Generate if-statement for each group of enables
        for req_key, enables in enables_by_requires.items():
            enable_lines = generate_grouped_enable_check(enables, member_map, ext_name_map)
            lines.extend(enable_lines)

        # Close function
        lines.extend([
            "    return false;",
            "}",
            "",
        ])

    # Generate dispatch map
    lines.extend([
        "// Capability dispatch map",
        "using CapCheckFn = bool(*)(uint32_t, const std::unordered_set<std::string>&,",
        "                           const VulkanFeatures&, const VulkanProperties&);",
        "",
        "// NOLINTNEXTLINE(cert-err58-cpp)",
        "const std::unordered_map<std::string_view, CapCheckFn> capability_check_map = {",
    ])

    for cap_name, func_name in sorted(map_entries):
        lines.append(f'    {{"{cap_name}", {func_name}}},')

    lines.extend([
        "};",
        "",
        "} // namespace",
        "",
    ])

    # Generate main dispatch function
    lines.extend([
        "bool is_spirv_capability_supported(",
        "    const char* capability,",
        "    uint32_t vk_api_version,",
        "    const std::unordered_set<std::string>& enabled_extensions,",
        "    const VulkanFeatures& features,",
        "    const VulkanProperties& properties) {",
        "",
        "    const auto it = capability_check_map.find(capability);",
        "    if (it == capability_check_map.end()) {",
        "        return false;",
        "    }",
        "    return it->second(vk_api_version, enabled_extensions, features, properties);",
        "}",
        "",
    ])

    return lines


def generate_capability_extensions_impl(
    capabilities: list[SpirvCapability],
    ext_name_map: dict[str, str],
) -> list[str]:
    """Generate get_spirv_capability_extensions() implementation using map lookup."""
    lines = [
        "namespace {",
        "struct CapExtEntry {",
        "    const char* ext;",
        "    uint32_t below_version;",
        "};",
        "",
        "// NOLINTNEXTLINE(cert-err58-cpp)",
        "const std::unordered_map<std::string_view, std::vector<CapExtEntry>> cap_ext_map = {",
    ]

    for cap in sorted(capabilities, key=lambda c: c.name):
        entries = []
        extensions_added = set()

        for enable in cap.enables:
            # Direct extension enables
            if enable.extension and enable.extension in ext_name_map:
                ext_macro = ext_name_map[enable.extension]
                if ext_macro not in extensions_added:
                    extensions_added.add(ext_macro)
                    entries.append((ext_macro, "0"))

            # Extensions from requires attribute
            if enable.requires:
                req_version, req_extensions = parse_requires(
                    enable.requires, ext_name_map
                )
                for ext_macro in req_extensions:
                    if ext_macro not in extensions_added:
                        extensions_added.add(ext_macro)
                        entries.append((ext_macro, req_version if req_version else "0"))

        if not entries:
            continue

        entry_strs = ", ".join(f"{{{ext}, {ver}}}" for ext, ver in entries)
        lines.append(f'    {{"{cap.name}", {{{entry_strs}}}}},')

    lines.extend(
        [
            "};",
            "} // namespace",
            "",
            "std::vector<const char*> get_spirv_capability_extensions(",
            "    const char* capability,",
            "    uint32_t vk_api_version) {",
            "",
            "    const auto it = cap_ext_map.find(capability);",
            "    if (it == cap_ext_map.end()) {",
            "        return {};",
            "    }",
            "    std::vector<const char*> result;",
            "    for (const auto& entry : it->second) {",
            "        if (entry.below_version == 0 || vk_api_version < entry.below_version) {",
            "            result.push_back(entry.ext);",
            "        }",
            "    }",
            "    return result;",
            "}",
            "",
        ]
    )

    return lines


def generate_capability_features_impl(
    capabilities: list[SpirvCapability], tags: list[str]
) -> list[str]:
    """Generate get_spirv_capability_features() implementation using map lookup."""
    lines = [
        "namespace {",
        "// NOLINTNEXTLINE(cert-err58-cpp)",
        "const std::unordered_map<std::string_view, std::vector<const char*>> cap_feature_map = {",
    ]

    for cap in sorted(capabilities, key=lambda c: c.name):
        feature_enables = [
            e for e in cap.enables if e.feature_struct and e.feature_name
        ]

        if not feature_enables:
            continue

        seen_features = set()
        feature_names = []
        for enable in feature_enables:
            if enable.feature_name in seen_features:
                continue
            seen_features.add(enable.feature_name)
            feature_names.append(f'"{enable.feature_name}"')

        feature_list = ", ".join(feature_names)
        lines.append(f'    {{"{cap.name}", {{{feature_list}}}}},')

    lines.extend(
        [
            "};",
            "} // namespace",
            "",
            "std::vector<std::string> get_spirv_capability_features(",
            "    const char* capability,",
            "    [[maybe_unused]] uint32_t vk_api_version) {",
            "",
            "    const auto it = cap_feature_map.find(capability);",
            "    if (it == cap_feature_map.end()) {",
            "        return {};",
            "    }",
            "    return {it->second.begin(), it->second.end()};",
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
    struct_map: dict[str, dict],
) -> str:
    """Generate the vulkan_spirv.cpp implementation file content."""
    lines = generate_file_header(VULKAN_SPEC_VERSION) + [
        '#include "merian/vk/utils/vulkan_spirv.hpp"',
        "",
        "#include <string_view>",
        "#include <unordered_map>",
        "#include <unordered_set>",
        "",
        "namespace merian {",
        "",
    ]

    lines.extend(generate_spirv_extensions_impl(extensions))
    lines.extend(generate_spirv_extension_requirements_impl(extensions, ext_name_map))
    lines.extend(generate_spirv_capabilities_impl(capabilities))
    lines.extend(generate_is_capability_supported_impl(capabilities, tags, struct_map, ext_name_map))
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

    print("Building property struct map...")
    struct_map = build_property_struct_map(xml_root, tags)
    print(f"Found {len(struct_map)} property structures")

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
    impl_content = generate_implementation(extensions, capabilities, tags, ext_name_map, struct_map)
    with open(out_path / "vulkan_spirv.cpp", "w") as f:
        f.write(impl_content)

    print("\nDone!")
    print(f"Generated {len(extensions)} SPIR-V extension entries")
    print(f"Generated {len(capabilities)} SPIR-V capability entries")
    print(f"Header lines: ~{len(header_content.splitlines())}")
    print(f"Implementation lines: ~{len(impl_content.splitlines())}")


if __name__ == "__main__":
    main()
