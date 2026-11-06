project(
  'tinyexr',
  'cpp',
  license: 'BSD-3-Clause',
  version: '3.2.0',
  meson_version: '>=0.64.0',
)

stb = dependency('stb', fallback: ['stb', 'stb_dep'], default_options: ['stb_libraries=[\'stb_image\', \'stb_image_write\']'])

include_dirs = include_directories('.')

args = ['-DTINYEXR_USE_MINIZ=0', '-DTINYEXR_USE_STB_ZLIB=1']

tinyexr_lib = static_library(
  'tinyexr',
  ['tinyexr_implementation.cpp'],
  cpp_args: args,
  include_directories: include_dirs,
  dependencies: [stb],
  pic: true,
  override_options: ['warning_level=0'],
  install: true,
)

tinyexr_dep = declare_dependency(
  compile_args: args,
  include_directories: include_dirs,
  link_with: tinyexr_lib,
  dependencies: [stb],
)
