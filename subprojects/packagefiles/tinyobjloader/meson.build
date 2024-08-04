project(
  'tinyobjloader',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().substring(1),
  meson_version: '>=0.64.0',
)

include_dirs = include_directories('.')

tol = library(
  'tol',
  ['tiny_obj_loader.cc'],
  include_directories: include_dirs,
  install: true,
)

tinyobjloader_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: tol,
)

tinyobjloader_headers_dep = declare_dependency(
  include_directories: include_dirs,
)
