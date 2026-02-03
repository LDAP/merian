"""Vulkan specification loading and parsing utilities."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen


def get_latest_vulkan_spec_version(use_tags: bool = True) -> str:
    """
    Get the latest Vulkan specification version from GitHub.

    Args:
        use_tags: If True, get latest tag (may include prereleases).
                  If False, get latest official release.

    Returns the latest tag/release name (e.g., 'v1.4.342').
    Falls back to a known version if the query fails.
    """
    try:
        if use_tags:
            # Query GitHub API for latest tag (most recent)
            api_url = "https://api.github.com/repos/KhronosGroup/Vulkan-Docs/tags"
            with urlopen(api_url) as response:
                data = json.loads(response.read())
                if data and len(data) > 0:
                    return data[0]['name']
        else:
            # Query GitHub API for latest release (stable)
            api_url = "https://api.github.com/repos/KhronosGroup/Vulkan-Docs/releases/latest"
            with urlopen(api_url) as response:
                data = json.loads(response.read())
                return data['tag_name']
    except Exception:
        # Fallback to a known version if GitHub API fails
        return "v1.4.338"

    # Should not reach here
    return "v1.4.338"


# Read version from environment variable, fallback to latest from GitHub
# Set VULKAN_SPEC_VERSION environment variable to override (e.g., "v1.4.338")
# By default, uses latest tag from GitHub (may be newer than latest release)
VULKAN_SPEC_VERSION = os.environ.get("VULKAN_SPEC_VERSION") or get_latest_vulkan_spec_version(use_tags=True)
VULKAN_SPEC_URL = f"https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/{VULKAN_SPEC_VERSION}/xml/vk.xml"

print(f"Using Vulkan spec version: {VULKAN_SPEC_VERSION}")


def load_vulkan_spec() -> ET.Element:
    """Download and parse the Vulkan XML specification."""
    with urlopen(VULKAN_SPEC_URL) as response:
        xml_root = ET.parse(response).getroot()
    return xml_root


def load_vendor_tags(xml_root: ET.Element) -> list[str]:
    """Extract vendor tags (KHR, EXT, etc.) from the Vulkan spec."""
    return [i.get("name") for i in xml_root.findall("tags/tag")]  # pyright: ignore[reportReturnType]


def build_skiplist(xml_root: ET.Element) -> set[str]:
    """
    Build a set of platform-specific and non-vulkan types to skip.

    This includes:
    - Types from platform-specific extensions (Android, Win32, etc.)
    - Types from non-vulkan API features (e.g., vulkansc)
    """
    skiplist = set()

    # Skip types from platform-specific extensions
    for ext in xml_root.findall("extensions/extension"):
        if ext.get("platform") is not None:
            for req in ext.findall("require"):
                for type_elem in req.findall("type"):
                    if name := type_elem.get("name"):
                        skiplist.add(name)

        # Skip types from extensions not supported on vulkan
        ext_supported = ext.get("supported", "")
        if "vulkan" not in ext_supported.split(","):
            for req in ext.findall("require"):
                for type_elem in req.findall("type"):
                    if name := type_elem.get("name"):
                        skiplist.add(name)

    # Skip types from non-vulkan API features (e.g., vulkansc)
    for feat in xml_root.findall("feature"):
        if (s := feat.get("api")) is not None and "vulkan" not in s.split(","):
            for req in feat.findall("require"):
                for type_elem in req.findall("type"):
                    if name := type_elem.get("name"):
                        skiplist.add(name)

    return skiplist


def get_output_paths() -> tuple[Path, Path]:
    """
    Return (implementation_path, include_path) for generated files.

    Paths are relative to the script directory.
    """
    script_dir = Path(__file__).parent.parent
    out_path = script_dir.parent / "src" / "merian" / "vk" / "utils"
    include_path = script_dir.parent / "include" / "merian" / "vk" / "utils"

    assert out_path.is_dir(), f"Output path does not exist: {out_path}"
    assert include_path.is_dir(), f"Include path does not exist: {include_path}"

    return out_path, include_path
