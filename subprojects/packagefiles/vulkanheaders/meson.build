project(
  'vulkanheaders',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().substring(1),
  meson_version: '>=0.64.0',
)

include_dirs = include_directories('include')

vulkanheaders_dep = declare_dependency(
  include_directories: include_dirs,
)
