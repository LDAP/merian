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

If a failure is preexisting, but the fix is quick, then fix instead of just report.

# Plugins

A plugin is a `shared_library('merian-plugin-<name>', name_prefix: '')` that consumes merian via
`dependency('merian')` and exports the `extern "C"` ABI hooks. Cloned into
`subprojects/merian-plugin-<name>`, it is auto-built as a merian subproject and discovered at
runtime without edits to merian's build files.

A plugin shares merian's ABI, so it **must** be built with merian's assertion/optimization config:
mixing `NDEBUG` / `_GLIBCXX_ASSERTIONS` / optimization across the `.so` boundary is undefined and
crashes deep in the driver. A plugin therefore must not pin `buildtype`/`b_ndebug` in its
`project()`. Plugins cloned into `subprojects/` inherit merian's config; standalone, pass `--buildtype`.

# Coding style

Expect the code to be read only by experienced (graphics) programmers.

## Comments

- Keep a minimal comment style; single-line, short, concise.
- Code should be self explaining, if it's not, maybe a variable as to be renamed or the code has to be rewritten, resort to an explaining comment only as last resort at a complex function that is not clear even to an experienced programmer (don't explain well known concepts).
- If a code or formula is from a paper, cite it.
- Don't mention implementation details to users of a function, class, or interface.
- Don't mention alternatives you did not implement (even if tested) or justifications.
- Don't explain where a symbol is used elsewhere (the call sites document that)
- Don't point to commit history, removed files, or the old implementation
- Don't reference concrete implementations (from an abstract class/interface). Don't comment about how
  subclasses might override, design alternatives considered, or future intent.
- Inside long methods, label sub-sections with one-liner comments (`// 1. ...`, `// upload prev
  vertices`) — never banner separators.
- File-level major dividers (`// --- Section ---`) are allowed sparingly for the obvious lifecycle
  splits (constructor / building / update). Don't multiply them.
- Keep reusable code domain-agnostic. A generic BSDF / shader util must not carry glTF (or any
  other spec's) function or variable names in its comments or identifiers — describe the math
  itself, with a bare citation if useful. The spec's vocabulary and parameter mapping belong only
  in the layer that implements that spec (e.g. the glTF material), never in the building blocks it
  composes.

## Naming

- Descriptive: `mesh_id`, `node_id`, `vertex_count`, `prim_count` — not `mid`, `nid`, `vc`.
- Tight-scope math locals can be terse (`m`, `it`, `v`, `pv`) but only when the surrounding code
  makes the role obvious.
- Lifecycle verb conventions: `add_*`, `upload_*`, `compute_*`, `build_*`,
  `ensure_*`. One verb per concept; pick one and stick to it.
- Match existing conventions in this codebase, not generic ones.

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
  never a raw pointer where a handle is expected.
- No global singletons for cross-cutting wiring. The graph already has an event system — use it
  (e.g. ImGui draws via a graph event, not a registry singleton).
- Header/source split: declarations in the `.hpp`, definitions in the `.cpp`. Keep includes
  alphabetical.

## Commits

- Subject: `<lib>: <component>: <summary>` (`merian-graph: accumulate: ...`, `merian-shaders: hash
  grid: ...`, `chore: ...`, `build: ...`). Most changes need nothing else.
- The body carries **only what the code cannot**: the symptom that motivated the change, measured
  numbers, a bare citation. Never restate the mechanism — that is what the comment at the code is
  for, and a body that duplicates it is noise.
- If available, quote measurements with the setup that produced them: `Quake mcpg 1280x720, RTX 5070 Laptop,
  timedemo demo1 over 500 frames render 12.27 -> 11.20 ms (-8.7 %)`.
- No verification narration, no provenance ("port of \<sha\>", "rebased onto"), no alternatives
  considered, no trailers of any kind (`Co-Authored-By`, session links).

## Process

Use clang-format on the modified files. Before a non-trivial design decision (replacing a
mechanism, introducing a new abstraction), ask back rather than committing to one direction.
