"""
merian: a Vulkan rendering framework.

Context is the Vulkan instance and device; graphs of nodes live in merian.graph.
"""

import sys


def _add_dll_directories() -> None:
    import os

    if os.name != "nt":
        return
    devenv = os.path.join(os.path.dirname(__file__), "devenv.env")
    if os.path.isfile(devenv):
        with open(devenv, encoding="utf-8") as f:
            for line in f:
                key, _, value = line.strip().partition("=")
                if key == "PATH":
                    os.environ["PATH"] = (
                        value.strip('"') + os.pathsep + os.environ.get("PATH", "")
                    )
    # Extension modules load DLLs only from here, not PATH; last added is searched first
    for directory in reversed(os.environ.get("PATH", "").split(os.pathsep)):
        if os.path.isabs(directory) and os.path.isdir(directory):
            os.add_dll_directory(directory)


_add_dll_directories()

from ._core import (  # noqa: E402
    DEBUG_BUILD,
    DEVICE_FAULT_EXTENSION,
    Context,
    MerianError,
    MissingExtension,
    ProfilerReport,
    VulkanError,
    io,
    plugins,
    set_log_level,
)
from .properties import Properties  # noqa: E402

sys.modules[f"{__name__}.io"] = io
sys.modules[f"{__name__}.plugins"] = plugins

__all__ = (
    # Context
    "DEBUG_BUILD",
    "DEVICE_FAULT_EXTENSION",
    "Context",
    # Errors
    "MerianError",
    "MissingExtension",
    "VulkanError",
    # Properties and profiling
    "ProfilerReport",
    "Properties",
    # Logging
    "set_log_level",
    # Submodules
    "io",
    "plugins",
)
