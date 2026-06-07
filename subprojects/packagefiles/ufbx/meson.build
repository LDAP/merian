project(
  'ufbx',
  'c',
  license: 'MIT',
  version: '0.9.0',
  meson_version: '>=0.64.0',
)

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

include_dirs = include_directories('.')

ufbx_lib = static_library(
  'ufbx',
  ['ufbx.c'],
  dependencies: [m_dep],
  include_directories: include_dirs,
  install: true,
)

ufbx_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: ufbx_lib,
  dependencies: [m_dep],
)
