project(
  'hlslpp',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().substring(1),
  meson_version: '>=0.64.0',
)

include_dirs = include_directories('include')

global_args = ['-DHLSLPP_FEATURE_TRANSFORM=1']
add_project_arguments(global_args, language: 'cpp')

hlslpp_dep = declare_dependency(
  compile_args: global_args,
  include_directories: include_dirs,
)
