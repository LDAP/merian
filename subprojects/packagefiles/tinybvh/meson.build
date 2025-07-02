project(
  'tinybvh',
  'cpp',
  license: 'MIT',
  version: '1.6.0',
  meson_version: '>=0.64.0',
)
fs = import('fs')
include_dirs = include_directories('.')

implementation = fs.copyfile('tiny_bvh.h', 'tiny_bvh.h.cpp')

tinybvh = static_library(
  'tinybvh',
  implementation,
  cpp_args: ['-DTINYBVH_IMPLEMENTATION'],
  include_directories: include_dirs,
  install: true,
)

tinybvh_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: tinybvh,
)

tinybvh_headers_dep = declare_dependency(
  include_directories: include_dirs,
)
