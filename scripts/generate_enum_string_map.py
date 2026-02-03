#!/usr/bin/env python3

from vulkan_codegen.codegen import generate_file_header
from vulkan_codegen.naming import remove_tag, to_camel_case, to_upper_case
from vulkan_codegen.spec import (
    VULKAN_SPEC_VERSION,
    build_skiplist,
    get_output_paths,
    load_vendor_tags,
    load_vulkan_spec,
)

xml = load_vulkan_spec()
tags = load_vendor_tags(xml)
skiplist = build_skiplist(xml)
out_path, _ = get_output_paths()

with open(out_path / "vk_enums.cpp", "w") as impl:
    # Write file header
    for line in generate_file_header(VULKAN_SPEC_VERSION):
        impl.write(line + "\n")

    impl.write(
        """\
#include "merian/utils/enums.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

"""
    )

    done = []
    for enum in xml.findall("enums"):
        if enum.get("type") != "enum":
            continue
        spec_enum_name = enum.get("name")
        assert spec_enum_name
        if spec_enum_name in skiplist:
            continue
        if spec_enum_name in done:
            continue
        tag = ""
        for s in tags:
            if spec_enum_name.endswith(s):
                tag = s
        enum_name = spec_enum_name.removeprefix("Vk")
        done.append(spec_enum_name)

        values = []
        for i in enum:
            value_name = i.get("name")
            if not value_name:
                continue
            if i.get("deprecated") is not None:
                continue
            if i.get("alias") is not None:
                continue
            value_name = remove_tag(value_name, tag)
            value_name = value_name.removeprefix(
                to_upper_case(spec_enum_name.removesuffix(tag))
            )
            value_name = value_name.removeprefix("VK_")
            value_name = to_camel_case(value_name, tags)
            value_name = f"VULKAN_HPP_NAMESPACE::{enum_name}::e" + value_name
            values.append(value_name)

        impl.write(
            f"static constexpr std::array<VULKAN_HPP_NAMESPACE::{enum_name}, {len(values)}> _{enum_name}_values = {{\n    {(',\n    '.join(values))}}};\n\n"
        )

        impl.write(
            f"""\
template<> uint32_t enum_size<VULKAN_HPP_NAMESPACE::{enum_name}>() {{
    return _{enum_name}_values.size();
}}

template<> const VULKAN_HPP_NAMESPACE::{enum_name}* enum_values<VULKAN_HPP_NAMESPACE::{enum_name}>() {{
    return _{enum_name}_values.data();
}}

template<> std::string enum_to_string<VULKAN_HPP_NAMESPACE::{enum_name}>(VULKAN_HPP_NAMESPACE::{enum_name} value) {{
    return vk::to_string(value);
}}

"""
        )

    impl.write("""
}
""")
