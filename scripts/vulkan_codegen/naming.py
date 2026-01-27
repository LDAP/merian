"""Name conversion utilities for Vulkan code generation."""


def to_camel_case(name: str, vendor_tags: list[str]) -> str:
    """
    Convert UPPER_SNAKE_CASE to CamelCase, matching vulkan.hpp conventions.

    Rules:
    - After underscore: next char stays uppercase
    - After digit: next char stays uppercase
    - Otherwise: lowercase
    - Vendor tags (KHR, EXT, etc.) stay uppercase at the end

    Args:
        name: The UPPER_SNAKE_CASE name to convert
        vendor_tags: List of vendor tags (e.g., ["KHR", "EXT", "AMD"])

    Returns:
        The CamelCase name with vendor tag preserved
    """
    tag = ""
    if (s := name.split("_")[-1]) in vendor_tags:
        tag = s
        name = name[: -1 - len(s)]

    result = ""
    last = None
    for c in name:
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


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case for member names."""
    result = ""
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            # Insert underscore before uppercase if previous char was lowercase or digit
            if name[i - 1].islower() or name[i - 1].isdigit():
                result += "_"
            # Insert underscore before uppercase if next char is lowercase (handles acronyms)
            elif i + 1 < len(name) and name[i + 1].islower():
                result += "_"
        result += c.lower()
    return result


def to_upper_case(name: str) -> str:
    """Convert CamelCase to UPPER_SNAKE_CASE."""
    result = ""
    prev_lower = False
    prev_digit = False

    for c in name:
        if c.isupper() and (prev_lower or prev_digit) or (c.isdigit() and prev_lower):
            result += "_"
        result += c.upper()
        prev_lower = c.islower()
        prev_digit = c.isdigit()

    return result


def vk_name_to_cpp_name(vk_name: str) -> str:
    """Convert VkFoo to Foo."""
    return vk_name.removeprefix("Vk")


def get_stype_from_name(name: str) -> str:
    """
    Convert a struct name to its sType enum value.

    e.g., VkPhysicalDevicePushDescriptorProperties ->
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES
    """
    result = "VK_STRUCTURE_TYPE_"
    prev_lower = False
    prev_digit = False

    for c in name.removeprefix("Vk"):
        if c.isupper() and (prev_lower or prev_digit):
            result += "_"
        elif c.isdigit() and prev_lower:
            result += "_"
        result += c.upper()
        prev_lower = c.islower()
        prev_digit = c.isdigit()

    return result


def generate_member_name(cpp_name: str) -> str:
    """
    Generate member variable name from struct name.

    PhysicalDevicePushDescriptorPropertiesKHR -> m_push_descriptor_properties_khr
    PhysicalDeviceProperties2 -> m_properties2
    """
    name = cpp_name.removeprefix("PhysicalDevice")
    snake = to_snake_case(name)
    return f"m_{snake}"


def generate_getter_name(cpp_name: str) -> str:
    """
    Generate getter method name from struct name.

    PhysicalDevicePushDescriptorPropertiesKHR -> get_push_descriptor_properties_khr
    PhysicalDeviceProperties2 -> get_properties_2
    """
    name = cpp_name.removeprefix("PhysicalDevice")
    snake = to_snake_case(name)
    return f"get_{snake}"


def remove_tag(with_tag: str, tag: str) -> str:
    """Remove vendor tag suffix from a name."""
    return with_tag.removesuffix(tag).removesuffix("_")
