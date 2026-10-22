# Merian 🎨

~ _A Vulkan development framework._ ~

<p align="middle">
  <img src="images/path_debugger.png" width="400" />
  <img src="images/graph_gui.png" width="400" />
</p>
<p align="middle">
  <img src="images/alk.png" width="200" />
  <img src="images/caustics.png" width="200" />
  <img src="images/ABeautifulGame.png" width="200" />
  <img src="images/tears.png" width="200" />
</p>

Merian is split into multiple components:

 - [`merian`](https://github.com/LDAP/merian/tree/main/include/merian): Provides core abstractions and utilities (Vulkan context, memory allocation, configuration, IO, ...).
 - [`merian-graph`](https://github.com/LDAP/merian/tree/main/include/merian-graph): Implements an extensible Vulkan processing graph. Already implemented nodes can be found [here](https://github.com/LDAP/merian/tree/main/include/merian-graph/nodes).
 - [`merian-shaders`](https://github.com/LDAP/merian/tree/main/include/merian-shaders): Reusable shader code plus the scene representation, glTF/FBX/PBRT loaders, material system and texture management.

Merian aims for compatibility with Windows, Linux as well as all major GPU vendors.

## Getting started

### Build

You need the [Meson](https://github.com/mesonbuild/meson) buildsystem along with ninja and cmake. You can install it using your favorite package manager or pip: `pip3 install meson ninja cmake`.

```bash
git clone --recursive https://github.com/LDAP/merian
cd merian
# optionally clone any plugin you want to use into subprojects:
# cd subprojects
# git clone https://github.com/LDAP/merian-plugin-quake
meson setup build
# You can enable / disable some feature by appending -Dfeature_name=enabled, for example glslang (GLSL compiler), glfw (windowing), sdl (windowing+audio), performance_profiling, tinygltf, ufbx, pbrt (scene formats) or switch to a debug build with --buildtype=debug/debugoptimized.
meson compile -C build
# this builds all the shared libraries and the merian Python package; run a graph (without one, an empty window):
python examples/graph_run.py
```

### Examples: graph_run.py

merian ships a generic runner [`graph_run.py`](https://github.com/LDAP/merian/tree/main/examples/graph_run.py) that loads a [`merian-graph`](https://github.com/LDAP/merian/tree/main/include/merian-graph) from a JSON file and runs it (`python examples/graph_run.py [options] [graph.json [args...]]`; without an argument it opens an empty window). It uses the `merian` Python package, whose build needs the Python headers (`-Dpython=disabled` builds merian without it). Options:

- `--loglevel=<trace|debug|info|warn|error>`: log verbosity.
- `--plugin-path=<dir>`: extra plugin search directory (repeatable).
- `--validation=<on|off|ifdebug>`: Vulkan validation layers (default: `ifdebug`).
- `--max-iterations=<n>`, `--exit-on-event=<pattern>`, `--time-delta=<ms>`: end the run or fix the time step.
- `--print-times`, `--print-memory`: profile the run, report the graph's memory.
- `--merge <file.json>`: merge a JSON fragment into the config (repeatable).
- `--set <pointer>=<value>`: assign a value at a JSON pointer, e.g. `--set /nodes/render/properties/x=1`.
- The options a graph declares: a choice from a preset folder (its `"presets"` key) or a value at a JSON pointer (its `"cli"` key); see the per-graph options below.
- `--help`: with a `graph.json`, also lists that graph's options.

It ships with a set of built-in nodes and can be extended with additional node sets and renderers through its [plugin system](#plugins).

Example graphs for `graph_run.py` are in the [`examples`](https://github.com/LDAP/merian/tree/main/examples) folder. Their windows show the image only: edit a graph while it runs with the [graph editor](https://github.com/LDAP/merian/tree/main/examples/graph_editor.py), or add the ImGui properties window with `--mixin imgui`.

- [`hdr_viewer.json`](https://github.com/LDAP/merian/tree/main/examples/hdr_viewer.json): a tone-mapped HDR image viewer — `python examples/graph_run.py examples/hdr_viewer.json <image.hdr>`.
- [`shadertoy.json`](https://github.com/LDAP/merian/tree/main/examples/shadertoy.json): runs a Shadertoy-style shader — `python examples/graph_run.py examples/shadertoy.json <shader.glsl>`.
- [`gltf.json`](https://github.com/LDAP/merian/tree/main/examples/gltf.json) / [`fbx.json`](https://github.com/LDAP/merian/tree/main/examples/fbx.json) / [`pbrt.json`](https://github.com/LDAP/merian/tree/main/examples/pbrt.json): a glTF / FBX / PBRTv4 scene viewer — `python examples/graph_run.py examples/gltf.json <scene.gltf> [options]`. Options:
    - `<scene>` (required): path to the glTF / FBX / PBRT scene to load.
    - `--renderer <pt|pt_mcpg|pt_ssmm|restir_di|restir_pt>`: renderer, one of [`examples/renderers`](https://github.com/LDAP/merian/tree/main/examples/renderers) (default: `pt`). Also accepts a path to a custom renderer fragment, e.g. `--renderer my/renderer.json`.
    - `--denoiser <none|svgf|dlss|none_volume|svgf_volume|dlss_volume>`: denoiser, one of [`examples/denoisers`](https://github.com/LDAP/merian/tree/main/examples/denoisers) (default: `none`, which accumulates). The `_volume` variants also trace the scene's volume and denoise it (path-traced renderers). `dlss` needs merian built with the NGX SDK (`-Ddlss`).
    - `--mixin <debug|imgui|...>`: additions from [`examples/mixins`](https://github.com/LDAP/merian/tree/main/examples/mixins), repeatable; `debug` attaches the path debugger, `imgui` adds the ImGui properties window (any example graph).
    - `--env-map <path>`: lat-long HDR environment map (sets the env type to `LatLong`).
    - `--max-path-length <n>`: maximum path length (path-traced renderers).
    - `--spp <n>`: samples per pixel.

### Examples: Python scripts

Besides `graph_run.py`, the `examples` folder holds scripts on the `merian` Python package; run from a source tree, they find the package of `build`, as `import merian` does in the repository root. Their few Python dependencies are dependency groups of `pyproject.toml`: `pip install --group scripts`, `--group gui` for the editor. The scene scripts take a scene (the graph follows its file type: glTF, FBX, PBRT) or a graph config, followed by that graph's options; `--help` lists both.

- [`scene_viewer.py`](https://github.com/LDAP/merian/tree/main/examples/scene_viewer.py): shows a scene in a window; the options of `graph_run.py` apply too.
- [`render_image.py`](https://github.com/LDAP/merian/tree/main/examples/render_image.py): renders a scene headless into an image (`--samples`, `--output`).
- [`render_reference.py`](https://github.com/LDAP/merian/tree/main/examples/render_reference.py): a reference image, accumulated in batches of distinct seeds; resumable.
- [`compare_methods.py`](https://github.com/LDAP/merian/tree/main/examples/compare_methods.py): compares variants against a reference: an image grid and a convergence plot.
- [`sweep_parameters.py`](https://github.com/LDAP/merian/tree/main/examples/sweep_parameters.py): renders every combination of values at JSON pointers, error and time per render.
- [`render_animation.py`](https://github.com/LDAP/merian/tree/main/examples/render_animation.py): renders a video, every frame at its own graph time.
- [`graph_editor.py`](https://github.com/LDAP/merian/tree/main/examples/graph_editor.py): a graph editor (PySide6): nodes and connections (right-click adds a node), the node's properties, the profiled times with their history, and where the memory goes; a preview, the backdrop and a panel of several views, each showing a window of the graph (its Window nodes present there and take the view's input, raw mouse included; set to SDL or GLFW, a Window node opens its own window instead), an output or a host sink, as an image or buffer values; play, pause and step, open and save configs; merian's log and the preferences. The graph runs on its own thread.
- [`image_grid.py`](https://github.com/LDAP/merian/tree/main/examples/image_grid.py): tiles images into a labelled contact sheet.

### Plugins

A plugin is a separate repository that builds a `merian-plugin-*` shared library and contributes nodes and/or context extensions, which are discovered automatically at startup by `graph_run.py` and any merian host. Plugins should never vendor merian but consume it via `dependency('merian')`. To build a plugin alongside merian, clone it into the `subprojects` folder (the directory must be named `merian-plugin-*`); `meson compile` then builds it as part of merian (no `PKG_CONFIG_PATH` needed):

```sh
git clone <plugin-repo> subprojects/merian-plugin-<name>
meson setup build --reconfigure
meson compile -C build
```

- [merian-plugin-quake](https://github.com/LDAP/Merian-plugin-quake): A scene node for Quake backed by the full power of quakespasm, including GUI support. 

## Environment variables

merian reads the following environment variables at startup:

| Variable | Default | Description |
| --- | --- | --- |
| `MERIAN_SHADER_CACHE` | on | Set to `0` to disable the on-disk Slang shader cache (serialized IR modules + compiled SPIR-V). |
| `MERIAN_SHADER_CACHE_DIR` | `./.merian-cache` | Directory for the shader cache. Safe to delete at any time. |
| `MERIAN_SHADER_CACHE_MAX_MB` | `512` | Cache size cap in MiB, enforced (least recently used first) once per process, when the first shader session ends. `0` = unbounded (manage by hand). |
| `MERIAN_TARGET_VK_API_VERSION` | highest supported | Target Vulkan API version, e.g. `1.3`. Clamped to the range supported by the Vulkan headers. |
| `MERIAN_DEFAULT_FILTER_VENDOR_ID` | — | Pick the GPU by PCI vendor id (decimal). |
| `MERIAN_DEFAULT_FILTER_DEVICE_ID` | — | Pick the GPU by device id (decimal). |
| `MERIAN_DEFAULT_FILTER_DEVICE_NAME` | — | Pick the GPU by (a substring of) its name. |
| `MERIAN_DEBUG_UTILS_ASSERT_ERROR` | on | When the `merian-debug-utils` extension is loaded, throw on a validation message of severity error. Set to `false` to only log it. |
| `MERIAN_PLUGIN_PATH` | — | Extra plugin search directories, separated by `:` (`;` on Windows). In addition, the user data dir is searched: `$XDG_DATA_HOME/merian/plugins` (or `$HOME/.local/share/merian/plugins`; `%APPDATA%\merian\plugins` on Windows). |

## Documentation

Documentation is in the `docs` subdirectory of this repository.

Nodes are documented in their [respective subfolder](https://github.com/LDAP/merian/tree/main/include/merian-graph/nodes).

## Integration in your own Project

If you do not want to use merians graph / plugin system, you can use merian in as a library in your own project. Examples:

- [merian-example-sum](https://github.com/LDAP/merian-example-sum) (computing a sum on the GPU) 
- [merian-quake](https://github.com/LDAP/merian-quake) (deprecated, if you want to play Quake use merian-plugin-quake).

Merian is similar to the `vulkan_raii.hpp` layer for `vulkan.hpp`. Most objects follow the RAII principle. The `merian` namespace provides shareable handles for most Vulkan types (e.g. `merian::ImageHandle` for `vk::Image`) and objects are automatically destroyed if their reference count becomes 0. The `Context` class initializes and destroys a Vulkan device and holds core objects (PhysicalDevice, Device, Queues, ...). Create a Context using `Context::create(ContextCreateInfo)`. The core `"merian"` extension is always loaded automatically. Make sure your program ends with `[INFO] [context.cpp:XX] context destroyed`. Note that the Vulkan dynamic dispatch loader must be used and the merian build system definition should already ensure that.

```c++
int main() {
    // "merian-resources" initializes a memory allocator, resource allocator and staging manager.
    // The core "merian" extension is always loaded automatically.
    merian::ContextHandle context = merian::Context::create({
        .context_extensions = {"merian-resources", "merian-debug-utils"},
        .application_name = "my-app",
    });

    auto alloc = context->get_context_extension<merian::ExtensionResources>()->resource_allocator();

    // allocating, rendering, ...

    // merian cleans up everything for you
}
```

To enable Vulkan extensions (e.g. ray tracing acceleration structures):

```c++
merian::ContextHandle context = merian::Context::create({
    .device_extensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME},
    .context_extensions = {"merian-resources"},
});
```

To enable Vulkan features (e.g. ray tracing):

```c++
merian::VulkanFeatures features;
features.set_feature("accelerationStructure", true);
features.set_feature("rayQuery", true);

merian::ContextHandle context = merian::Context::create({
    .features = features,
    .context_extensions = {"merian-resources", "GLFW"},
});
```

### Buildsystem Integration

This library uses the [Meson Build System](https://mesonbuild.com/) and declares a dependency for it:

``` py
# in your meson.build

merian = dependency('merian', version: ['>=2.0.0'], fallback: ['merian', 'merian_dep'])
# or if you want to use the shader compiler from merian
merian_subp = subproject('merian', version: ['>=2.0.0'])
merian = merian_subp.get_variable('merian_dep')
shader_generator = merian_subp.get_variable('shader_generator')

sources = ['app.cpp']
sources += shader_generator.process('shader0.comp', 'shader1.comp',...)
# or to prevent conflicts:
sources += shader_generator.process(
    'shader0.comp', 'shader1.comp',...
    extra_args: ['--prefix', 'my_app_c_prefix'],
    preserve_path_from: meson.source_root(),
)

exe = executable(
    'my-app',
    sources,
    dependencies: [
        merian,
    ],
    # ...
)
```

To allow meson to find Merian, either clone this repo into the `subprojects` folder or add a file `subprojects/merian.wrap` with

```ini
[wrap-git]
directory = merian

url = https://github.com/LDAP/merian
revision = 2.0.0
depth = 1
clone-recursive = true

[provide]
merian = merian_dep
```
