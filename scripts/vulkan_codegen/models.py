"""Data models for Vulkan code generation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtensionDep:
    """Represents a dependency that can be either an extension or a Vulkan version."""

    extension: Optional[str] = None  # e.g., VK_KHR_SWAPCHAIN_EXTENSION_NAME
    version: Optional[str] = None  # e.g., VK_API_VERSION_1_1


@dataclass
class Extension:
    """Represents a Vulkan extension."""

    name: str  # e.g., VK_KHR_swapchain
    name_macro: str  # e.g., VK_KHR_SWAPCHAIN_EXTENSION_NAME
    dependencies: list[list[ExtensionDep]] = field(default_factory=list)  # OR of ANDs


@dataclass
class PropertyStruct:
    """Represents a Vulkan property structure that extends VkPhysicalDeviceProperties2."""

    vk_name: str  # e.g., VkPhysicalDevicePushDescriptorPropertiesKHR
    cpp_name: str  # e.g., PhysicalDevicePushDescriptorPropertiesKHR
    stype: str  # e.g., VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR
    extension: Optional[str] = None  # e.g., VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
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
    extension: Optional[str] = None  # e.g., VK_EXT_ROBUSTNESS_2_EXTENSION_NAME
    promotion_version: Optional[str] = None  # e.g., VK_API_VERSION_1_2
    member_prefix: str = ""  # e.g., "features." for VkPhysicalDeviceFeatures2
    aliases: list[str] = field(default_factory=list)
