project(
  'miniply',
  'cpp',
  license: 'MIT',
  version: '1.0',
  meson_version: '>=0.64.0',
)

include_dirs = include_directories('.')

miniply_lib = static_library(
  'miniply',
  ['miniply.cpp'],
  include_directories: include_dirs,
  install: true,
)

miniply_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: miniply_lib,
)
