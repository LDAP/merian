# Project

merian is a Vulkan C++ rendering framework. It builds `merian-core` (shared library), the
`merian-graph` node/graph library, and the generic `merian-graph-run` executable. Renderers,
scenes, and other node sets live in separate repos as plugins (`merian-plugin-*`), not in this tree.

Use the same bash commands to prevent re-approval.

# Build & Run

Uses Meson.

Setup: run once if `build` does not exist: `meson setup build`
Compile: `meson compile -C build`
Run a graph: `build/merian-graph-run <config.json>`, better with timeout:
`timeout 15 build/merian-graph-run <config.json>`

# Tests

Enable tests: `meson configure build -Dtests=true`
Run one: `build/tests/test-<name>` (e.g. `test-small-vector`)

# Plugins

A plugin is a `shared_library('merian-plugin-<name>', name_prefix: '')` that consumes merian via
`dependency('merian')` and exports the `extern "C"` ABI hooks. Cloned into `plugins/<name>`, it is
auto-built as a merian subproject and discovered at runtime — no edits to merian's build files.

A plugin shares merian's ABI, so it **must** be built with merian's assertion/optimization config:
mixing `NDEBUG` / `_GLIBCXX_ASSERTIONS` / optimization across the `.so` boundary is undefined and
crashes deep in the driver. A plugin therefore must not pin `buildtype`/`b_ndebug` in its
`project()` — built in-tree it inherits merian's config; standalone, pass `--buildtype`. Per-
subproject option values are sticky, so after changing merian's buildtype use `meson setup --wipe`,
not `--reconfigure`, for plugins to follow.

# Coding style

Expect the code to be read only by experienced (graphics) programmers.

## Comments

- Single short line is the default; multi-line walls of text are out. If the explanation needs a
  paragraph, the code probably needs restructuring instead.
- Explain *why*, never *what*. Identifier names already say what. Don't mention implementation
  details to users of a function, class, or interface.
- Don't explain usage of well-known concepts (a type alias already makes a one-line-swap obvious).
- Inside long methods, label sub-sections with one-liner comments (`// 1. ...`, `// upload prev
  vertices`) — never banner separators.
- File-level major dividers (`// --- Section ---`) are allowed sparingly for the obvious lifecycle
  splits (constructor / building / update). Don't multiply them.
- Comments describe what the code does locally — the math, the invariant, the units, the
  non-obvious choice. They are not a place for cross-references or provenance narration. Do **not**
  explain where a symbol is used elsewhere (the call sites document that), narrate provenance, or
  justify a choice against alternatives. A bare citation (paper name/year, URL) is welcome; the
  chain of how the snippet got here is not.
- Drop comments that point to commit history, removed files, or the old pipeline, or that reference
  a concrete implementation from an abstract class/interface. Do **not** comment about how
  subclasses might override, design alternatives considered, or future intent.
- Keep reusable code domain-agnostic. A generic BSDF / shader util must not carry glTF (or any
  other spec's) function or variable names in its comments or identifiers — describe the math
  itself, with a bare citation if useful. The spec's vocabulary and parameter mapping belong only
  in the layer that implements that spec (e.g. the glTF material), never in the building blocks it
  composes.

## Naming

- Descriptive: `mesh_id`, `node_id`, `vertex_count`, `prim_count` — not `mid`, `nid`, `vc`.
- Tight-scope math locals can be terse (`m`, `it`, `v`, `pv`) but only when the surrounding code
  makes the role obvious.
- Lifecycle verb conventions: `add_*`, `mark_*_dirty`, `upload_*`, `compute_*`, `build_*`,
  `ensure_*`. One verb per concept; pick one and stick to it.
- Match existing conventions in this codebase, not generic ones: a connector is `out`, not
  `output`; a `properties()` parameter is `config`, not `props`.

## Code

- `const` on every local that isn't reassigned.
- Modern containers / idioms: `try_emplace`, `extract`, structured bindings, `auto [it, inserted]`,
  `assign(n, value)`, `std::move` on heavy types only.
- Replace hand-rolled matrix building with the library: `mul`, `transpose`, `inverse`, `identity`,
  `translation`, `scale`, `rotation`. Never write a 3-line "AngleVectors then fix-up" snippet.
- Use `enum` / `enum class` over magic constants.
- Prefer `std::unordered_map` over `std::map` unless iteration order matters.
- `static_cast`, never C-style casts.
- Ownership is expressed in the type: own with `std::shared_ptr`, observe with `std::weak_ptr` —
  never a raw pointer where a handle is expected. A controller that drives a camera owns a
  `CameraHandle`; a node observing a registered controller holds a `weak_ptr`.
- No global singletons for cross-cutting wiring. The graph already has an event system — use it
  (e.g. ImGui draws via a graph event, not a registry singleton). ImGui has no business inside the
  `Graph` core.
- Header/source split: declarations in the `.hpp`, definitions in the `.cpp`. Keep includes
  alphabetical.
- In `properties()`, render each control inside the method that owns it; don't wrap a node's
  config at the call site. Group nested settings with `st_begin_child(id, label)` using the default
  flag — `FRAMED` makes a child look different from its siblings.

## Process

Use clang-format on the modified files. Before a non-trivial design decision (replacing a
mechanism, introducing a new abstraction), ask back rather than committing to one direction.
