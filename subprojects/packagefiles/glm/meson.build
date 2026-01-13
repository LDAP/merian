project(
  'glm',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout(),
  meson_version: '>=0.64.0',
)

global_args = []

glm_dep = declare_dependency(
  compile_args: global_args,
  include_directories: include_directories('.')
)

