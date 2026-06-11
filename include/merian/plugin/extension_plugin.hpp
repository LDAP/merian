#pragma once

#include "merian/plugin/plugin_export.hpp"

#include <cstdint>

// Context-extension plugin ABI. Loaded by merian-core during Context creation. A plugin that
// provides context extensions implements these symbols; one that only provides graph nodes does
// not (see merian-graph/plugin/node_plugin.hpp).

namespace merian {
class ExtensionRegistry;
} // namespace merian

// Incremented on an incompatible change to this ABI or any core API a plugin links against. Core
// refuses to load an extension plugin reporting a different value.
#define MERIAN_EXTENSION_PLUGIN_ABI_VERSION 1u

extern "C" {

// Mandatory if merian_register_extensions is exported. Must return
// MERIAN_EXTENSION_PLUGIN_ABI_VERSION of the headers the plugin was built against.
MERIAN_PLUGIN_EXPORT uint32_t merian_extension_plugin_abi_version(void);

// Register context extensions with the host registry. Called once, before context creation.
MERIAN_PLUGIN_EXPORT void merian_register_extensions(merian::ExtensionRegistry& registry);

using merian_extension_plugin_abi_version_fn = uint32_t (*)();
using merian_register_extensions_fn = void (*)(merian::ExtensionRegistry&);

} // extern "C"
