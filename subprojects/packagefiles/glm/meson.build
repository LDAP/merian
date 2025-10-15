project(
  'glm',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout(),
  meson_version: '>=0.64.0',
)

global_args = [
  # Vulkan uses [0, 1] for depth instead of [-1, 1] like OpenGL
  '-DGLM_FORCE_DEPTH_ZERO_TO_ONE',
  '-DGLM_FORCE_SWIZZLE=1'
]

glm_dep = declare_dependency(
  compile_args: global_args,
  include_directories: include_directories('.')
)

