project(
  'ngx',
  'cpp',
  license: 'LicenseRef-NvidiaProprietary',
  version: '310.9.1',
  meson_version: '>=0.64.0',
)

ccx = meson.get_compiler('cpp')

if host_machine.cpu_family() != 'x86_64'
  error('the NGX SDK only ships x86_64 binaries.')
endif

include_dirs = include_directories('include')
lib_root = meson.current_source_dir() / 'lib'

if host_machine.system() == 'windows'
  lib_name = 'nvsdk_ngx_d'
  if ccx.get_argument_syntax() == 'msvc'
    # _d links the dynamic CRT (/MD), _s the static one (/MT); _dbg the debug flavour of either.
    # Mixing flavours is a link error, so mirror what meson passes: of the derived settings only
    # 'debug' takes the debug CRT, and 'plain' passes no flag at all, leaving cl's static default.
    vscrt = get_option('b_vscrt')
    if vscrt in ['from_buildtype', 'static_from_buildtype']
      buildtype = get_option('buildtype')
      debug_crt = buildtype == 'debug'
      static_crt = vscrt == 'static_from_buildtype' or buildtype == 'plain'
    else
      debug_crt = vscrt in ['mdd', 'mtd']
      static_crt = vscrt in ['mt', 'mtd']
    endif
    lib_name = static_crt ? 'nvsdk_ngx_s' : 'nvsdk_ngx_d'
    lib_name = debug_crt ? lib_name + '_dbg' : lib_name
  endif

  ngx_libs = [ccx.find_library(lib_name, dirs: lib_root / 'Windows_x86_64' / 'x64')]
  snippet_dir = lib_root / 'Windows_x86_64' / 'rel'
  snippets = ['nvngx_dlss.dll', 'nvngx_dlssd.dll']
else
  ngx_libs = [
    ccx.find_library('nvsdk_ngx', dirs: lib_root / 'Linux_x86_64'),
    ccx.find_library('dl', required: false),
  ]
  snippet_dir = lib_root / 'Linux_x86_64' / 'rel'
  snippets = [
    'libnvidia-ngx-dlss.so.' + meson.project_version(),
    'libnvidia-ngx-dlssd.so.' + meson.project_version(),
  ]
endif

# NGX loads the feature libraries from the executable's directory; installing them there covers the
# installed case, handing NGX the source path covers running from the build tree.
foreach snippet : snippets
  install_data(snippet_dir / snippet, install_dir: get_option('bindir'))
endforeach

ngx_dep = declare_dependency(
  compile_args: [
    '-DMERIAN_NGX_SNIPPET_DIR="@0@"'.format(snippet_dir.replace('\\', '/')),
  ],
  include_directories: include_dirs,
  dependencies: ngx_libs,
)
