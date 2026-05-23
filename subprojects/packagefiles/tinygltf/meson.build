project(
  'tinygltf',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().substring(1),
  meson_version: '>=0.64.0',
)

stb = dependency('stb', fallback: ['stb', 'stb_dep'],
                 default_options: ['stb_libraries=[\'stb_image\', \'stb_image_write\']'])

include_dirs = include_directories('.')

implementation = import('fs').copyfile('tiny_gltf.h', 'tiny_gltf.cpp')

tinygltf_lib = static_library(
  'tinygltf',
  [implementation],
  cpp_args: ['-DTINYGLTF_IMPLEMENTATION'],
  dependencies: [stb],
  include_directories: include_dirs,
  install: true,
)

tinygltf_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: tinygltf_lib,
  dependencies: [stb],
)
