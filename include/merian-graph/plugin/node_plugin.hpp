#pragma once

#include "merian/plugin/plugin_export.hpp"

#include <cstdint>

// Graph-node plugin ABI. Loaded by the merian-graph extension once it is active. A plugin that
// provides graph nodes implements these symbols; one that only provides context extensions does
// not (see merian/plugin/extension_plugin.hpp).

namespace merian {
class NodeRegistry;
} // namespace merian

// Incremented on an incompatible change to this ABI or any merian-graph API a plugin links
// against. The merian-graph extension refuses to load a node plugin reporting a different value.
#define MERIAN_NODE_PLUGIN_ABI_VERSION 1u

extern "C" {

// Mandatory if merian_register_nodes is exported. Must return MERIAN_NODE_PLUGIN_ABI_VERSION of
// the headers the plugin was built against.
MERIAN_PLUGIN_EXPORT uint32_t merian_node_plugin_abi_version(void);

// Register node types with the host registry. Called once, before context creation completes.
MERIAN_PLUGIN_EXPORT void merian_register_nodes(merian::NodeRegistry& registry);

using merian_node_plugin_abi_version_fn = uint32_t (*)();
using merian_register_nodes_fn = void (*)(merian::NodeRegistry&);

} // extern "C"
