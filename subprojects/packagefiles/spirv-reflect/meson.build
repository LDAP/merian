project(
  'spirv-reflect',
  'c',
  license: 'Apache License 2.0',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().replace('vulkan-sdk-', ''),
  meson_version: '>=0.64.0',
)

include_dirs = include_directories('.')

src = files(
  'spirv_reflect.c',
)

spirv_reflect = library(
  'spirv-reflect',
  src,
  include_directories: include_dirs,
  install: true,
)

spirv_reflect_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: spirv_reflect,
)
