# Plugins

Clone merian plugins here. A plugin is a git repository whose `meson.build` builds a
`merian-plugin-*` shared library against `dependency('merian')`.

```sh
git clone <plugin-repo> plugins/<name>
meson compile -C build
```

Every plugin found here is built as part of merian's own build (it resolves merian in-tree via
`meson.override_dependency`, so no `PKG_CONFIG_PATH` is needed) and is discovered automatically at
startup by `merian-graph-run` and any merian host.
