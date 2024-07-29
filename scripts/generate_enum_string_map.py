import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

VULKAN_SPEC_VERSION = "v1.3.275"
VULKAN_SPEC_URL = f"https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/{VULKAN_SPEC_VERSION}/xml/vk.xml"
VULKAN_ENUMS_URL = f"https://github.com/KhronosGroup/Vulkan-Headers/blob/{VULKAN_SPEC_VERSION}/include/vulkan/vulkan_enums.hpp"

out_path = Path(__file__).parent.parent / "src" / "merian" / "vk" / "utils"
assert out_path.is_dir()

with urlopen(VULKAN_SPEC_URL) as response:
    xml = ET.parse(response).getroot()

# with urlopen(VULKAN_ENUMS_URL) as response:
#     vulkan_enums = str(response.read())

tags = [i.get("name") for i in xml.findall("tags/tag")]

skiplist = []

for ext in xml.findall("extensions/extension"):
    if ext.get("platform") is None:
        continue
    for req in ext.findall("require"):
        for type in req.findall("type"):
            skiplist.append(type.get("name"))

for feat in xml.findall("feature"):
    if (s := feat.get("api")) is not None and "vulkan" not in s.split(","):
        for req in feat.findall("require"):
            for type in req.findall("type"):
                skiplist.append(type.get("name"))


def to_camel_case(snake_str, enum_tag):
    snake_str = snake_str.removesuffix(enum_tag).removesuffix("_")
    tag = ""
    if (s := snake_str.split("_")[-1]) in tags:
        tag = s
        snake_str = snake_str[: -1 - len(s)]

    result = ""
    last = None
    for c in snake_str:
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


with open(out_path / "vk_enums.cpp", "w") as impl:
    impl.write(
        f"""\
// This file was is autogenerated for Vulkan {VULKAN_SPEC_VERSION}.
// Created: {datetime.datetime.now()}

#include "enums.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {{

"""
    )

    done = []
    for enum in xml.findall("enums"):
        if enum.get("type") != "enum":
            continue
        enum_name = enum.get("name")
        assert enum_name
        if enum_name in skiplist:
            continue
        enum_name = enum_name[2:]
        if enum_name in done:
            continue
        tag = ""
        for s in tags:
            if enum_name.endswith(s):
                tag = s
        done.append(enum_name)

        values = []
        for i in enum:
            value_name = i.get("name")
            if not value_name:
                continue
            if i.get("deprecated") is not None:
                continue
            if i.get("alias") is not None:
                continue
            value_name = to_camel_case(value_name, tag)
            value_name = value_name.removeprefix("Vk")
            value_name = value_name.removeprefix(enum_name.removesuffix(tag))
            value_name = f"VULKAN_HPP_NAMESPACE::{enum_name}::e" + value_name
            values.append(value_name)

        impl.write(
            f"static constexpr std::array<VULKAN_HPP_NAMESPACE::{enum_name}, {len(values)}> _{enum_name}_values = {{\n    {(',\n    '.join(values))}}};\n\n"
        )

        # print(enum.get("name"))
        # for i in enum:
        # print(i.get("value"))
        # print(i.get("name"))

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