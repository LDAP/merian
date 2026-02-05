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
class VulkanStruct:
    """Base class for Vulkan struct metadata extracted from XML."""

    vk_name: str  # e.g., VkPhysicalDeviceRobustness2FeaturesEXT
    cpp_name: str  # e.g., PhysicalDeviceRobustness2FeaturesEXT
    stype: str  # e.g., VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT
    extension_name: Optional[str] = None  # e.g., VK_EXT_robustness_2 (reference to Extension)
    structextends: list[str] = field(default_factory=list)  # e.g., ["VkPhysicalDeviceFeatures2"]
    aliases: list[str] = field(default_factory=list)  # list of alias names pointing to this struct
    is_alias: bool = False  # True if this struct is an alias of another
    members: list[tuple[str, str, str]] = field(default_factory=list)  # (type, name, comment) for ALL members


@dataclass
class FeatureMember:
    """Represents a VkBool32 member in a feature structure."""

    name: str
    comment: str = ""


@dataclass
class FeatureStruct(VulkanStruct):
    """Represents a Vulkan feature structure that extends VkPhysicalDeviceFeatures2."""

    feature_members: list[FeatureMember] = field(default_factory=list)  # Only VkBool32 members
    required_version: Optional[str] = None  # e.g., VK_API_VERSION_1_3 (for core VulkanXXFeatures)
    feature_member_prefix: str = ""  # e.g., "features." for VkPhysicalDeviceFeatures2


@dataclass
class PropertyStruct(VulkanStruct):
    """Represents a Vulkan property structure that extends VkPhysicalDeviceProperties2."""

    core_version: Optional[str] = None  # e.g., VK_API_VERSION_1_4 (when it became core)
