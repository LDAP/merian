project(
  'stb',
  'c',
  license: 'MIT',
  version: '1.0.0',
  meson_version: '>=0.64.0',
)
fs = import('fs')
include_dirs = include_directories('.')

implementations = []
args = []

foreach lib : get_option('stb_libraries')
  header = lib + '.h'
  implementation = lib + '.c'
  if fs.exists(header)
    implementations += fs.copyfile(header, implementation)
    args += '-D' + lib.to_upper() + '_IMPLEMENTATION'
  endif

  if fs.exists(implementation)
    implementations += implementation
  endif
endforeach

stb = static_library(
  'stb',
  implementations,
  cpp_args: args,
  include_directories: include_dirs,
  install: true,
)

stb_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: stb,
)

stb_headers_dep = declare_dependency(
  include_directories: include_dirs,
)
