"""Data models for Vulkan code generation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Extension:
    """Represents a Vulkan extension."""

    name: str  # e.g., VK_KHR_swapchain
    name_macro: str  # e.g., VK_KHR_SWAPCHAIN_EXTENSION_NAME
    type: str  # device or instance
    extension_number: Optional[int] = None  # e.g., 81 (higher = newer extension)
    dependencies: list[str] = field(default_factory=list)  # e.g., ["VK_KHR_swapchain"] (flattened expression, only extensions)
    promotedto: Optional[str] = None  # e.g., VK_API_VERSION_1_3
    deprecatedby: Optional[str] = None  # e.g., VK_KHR_buffer_device_address, VK_API_VERSION_1_3
    property_types: list[str] = field(default_factory=list)  # e.g., ["ePhysicalDeviceProperties2"]
    feature_types: list[str] = field(default_factory=list)  # e.g., ["ePhysicalDeviceFeatures2"]


@dataclass
class PropertyStruct:
    """Represents a Vulkan property structure that extends VkPhysicalDeviceProperties2."""

    vk_name: str  # e.g., VkPhysicalDevicePushDescriptorPropertiesKHR
    cpp_name: str  # e.g., PhysicalDevicePushDescriptorPropertiesKHR
    stype: str  # e.g., VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR
    extension_name: Optional[str] = None  # e.g., VK_KHR_push_descriptor (reference to Extension)
    core_version: Optional[str] = None  # e.g., VK_API_VERSION_1_4 (when it became core)


@dataclass
class FeatureMember:
    """Represents a VkBool32 member in a feature structure."""

    name: str
    comment: str = ""


@dataclass
class FeatureStruct:
    """Represents a Vulkan feature structure that extends VkPhysicalDeviceFeatures2."""

    vk_name: str  # e.g., VkPhysicalDeviceRobustness2FeaturesEXT
    cpp_name: str  # e.g., PhysicalDeviceRobustness2FeaturesEXT
    stype: str  # e.g., VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT
    members: list[FeatureMember] = field(default_factory=list)
    extension_name: Optional[str] = None  # e.g., VK_EXT_robustness_2 (reference to Extension)
    required_version: Optional[str] = None  # e.g., VK_API_VERSION_1_3 (for core VulkanXXFeatures)
    member_prefix: str = ""  # e.g., "features." for VkPhysicalDeviceFeatures2
    aliases: list[str] = field(default_factory=list)
    structextends: list[str] = field(default_factory=list)  # e.g., ["VkPhysicalDeviceFeatures2", "VkDeviceCreateInfo"]
