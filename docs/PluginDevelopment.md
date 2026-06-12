# Plugin Development

A plugin is a shared library named `merian-plugin-<name>` that contributes nodes and/or context
extensions to a merian host (such as `merian-graph-run`). Plugins are discovered at startup, so a
new renderer or node set appears without rebuilding or relinking the host.

Plugins **do not vendor merian** — they consume it through `dependency('merian')` and link against
the installed (or in-tree) merian shared libraries.

## ABI entry points

A plugin exports a small set of `extern "C"` hooks; the host resolves them by name and calls those
that are present. Decorate each with `MERIAN_PLUGIN_EXPORT`.

| Hook | Header | Purpose |
| ---- | ------ | ------- |
| `merian_plugin_name()` | `merian/plugin/plugin_export.hpp` | Optional. Name used in logs. |
| `merian_node_plugin_abi_version()` + `merian_register_nodes(NodeRegistry&)` | `merian-graph/plugin/node_plugin.hpp` | Register node types. |
| `merian_extension_plugin_abi_version()` + `merian_register_extensions(ExtensionRegistry&)` | `merian/plugin/extension_plugin.hpp` | Register context extensions. |

Each `*_abi_version()` is mandatory if the matching `register` hook is exported and must return the
`MERIAN_*_PLUGIN_ABI_VERSION` of the headers the plugin was built against; the host refuses to load
a plugin whose version does not match.

A node plugin therefore looks like:

```cpp
#include "merian-graph/graph/node_registry.hpp"
#include "merian-graph/plugin/node_plugin.hpp"

extern "C" {

MERIAN_PLUGIN_EXPORT const char* merian_plugin_name(void) {
    return "my-plugin";
}

MERIAN_PLUGIN_EXPORT uint32_t merian_node_plugin_abi_version(void) {
    return MERIAN_NODE_PLUGIN_ABI_VERSION;
}

MERIAN_PLUGIN_EXPORT void merian_register_nodes(merian::NodeRegistry& registry) {
    registry.register_node_type<MyNode>("My Node", "Does something useful.");
}

} // extern "C"
```

Code that must run after the Vulkan context exists registers a `ContextExtension` via
`merian_register_extensions` instead — the host calls its lifecycle callbacks during context
bring-up.

## meson.build

```py
project('merian-plugin-myplugin', 'cpp', default_options: ['cpp_std=c++23'])

merian = dependency('merian')

shared_library(
    'merian-plugin-myplugin',
    'src/plugin.cpp',
    dependencies: merian,
    name_prefix: '',        # the file must be literally merian-plugin-myplugin.so
    install: true,
    install_dir: join_paths(get_option('libdir'), 'merian', 'plugins'),
)
```

Resources (runtime-compiled shaders, assets) go in a `res/` folder installed next to the `.so`;
merian's loader adds that directory to its resource search paths automatically.

## Building in-tree

Clone the plugin into merian's `plugins/` folder. `meson compile` then builds it as part of
merian — it resolves merian in-tree (via `meson.override_dependency`), so no `PKG_CONFIG_PATH` is
needed and the host picks it up at the next start:

```sh
git clone <plugin-repo> plugins/myplugin
meson compile -C build
```

## Building standalone (without vendoring merian)

The plugin is its own meson project; build it on its own against a merian that is either installed
system-wide or present as a build tree — without copying merian's sources into the plugin.

Against an installed merian, pkg-config finds it directly:

```sh
meson setup build
meson compile -C build
```

Against a merian build tree, point pkg-config at its uninstalled metadata:

```sh
PKG_CONFIG_PATH=<merian>/build/meson-uninstalled meson setup build
meson compile -C build
```

## ABI compatibility

A plugin shares merian's C++ ABI, so it **must** be built with the same assertion/optimization
configuration — mixing `NDEBUG` / `_GLIBCXX_ASSERTIONS` / optimization across the library boundary
is undefined and crashes at runtime.

Therefore a plugin must **not** pin `buildtype` or `b_ndebug` in its `project()` default options:

- built in-tree it inherits merian's configuration automatically;
- built standalone, pass `--buildtype` (and any `-Db_ndebug`) matching the merian you link against.

Per-subproject options are sticky, so after changing merian's `buildtype` run `meson setup --wipe`
(not `--reconfigure`) so in-tree plugins follow the new configuration.
