project('slang',
        'cpp',
        version: '2025.14.3')

fs = import('fs')
python = import('python').find_installation()

slang_version_fmt = 'slang-@0@-@1@-@2@'
slang_url_format = 'https://github.com/shader-slang/slang/releases/download/v@0@/'

if host_machine.system() == 'windows'
  system = 'windows'
elif host_machine.system() == 'darwin'
  system = 'macos'
elif host_machine.system() == 'linux'
  system = 'linux'
else
  error('Unsupported platform for Slang prebuilt binaries')
endif

if host_machine.cpu_family() not in ['aarch64', 'x86_64']
  error('Unsupported cpu family for Slang prebuilt binaries')
endif

slang_version = slang_version_fmt.format(meson.project_version(), system, host_machine.cpu_family())
slang_zip = slang_version + '.tar.gz'
slang_debug_info_zip = slang_version + '-debug-info.tar.gz'
slang_url = slang_url_format.format(meson.project_version()) + slang_zip
slang_debug_info_url = slang_url_format.format(meson.project_version()) + slang_debug_info_zip

slang_download_path = meson.current_build_dir() / slang_zip
slang_debug_info_download_path = meson.current_build_dir() / slang_debug_info_zip
extract_dir = slang_version

## DOWNLOAD SLANG (and debug-info)
#-------------------------------------------------

download_script = '''
import sys, urllib.request, pathlib
url, out = sys.argv[1], pathlib.Path(sys.argv[2])
if not out.exists():
    print(f"Downloading {url} -> {out}")
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        f.write(r.read())
else:
    print(f"{out} already exists, skipping download")
'''
if not fs.exists(slang_download_path)
  message('downloading slang from @0@'.format(slang_url))
  run_command(python, '-c', download_script, slang_url, slang_download_path, check: true)
else
  message('skipping download and using @0@'.format(slang_download_path))
endif

# if get_option('buildtype').startswith('debug')
#   if not fs.exists(slang_debug_info_download_path)
#     message('downloading slang debug info from @0@'.format(slang_debug_info_url))
#     run_command(python, '-c', download_script, slang_debug_info_url, slang_debug_info_download_path, check: true)
#   else
#     message('skipping download and using @0@'.format(slang_debug_info_download_path))
#   endif
# endif

## PREPARE EXTRACT DIRECTORY
#-------------------------------------------------

abs_extract_dir = meson.current_source_dir() / extract_dir

rmdir_script = '''
import sys, pathlib, shutil
target = pathlib.Path(sys.argv[1])
shutil.rmtree(target)
'''

# if fs.exists(abs_extract_dir)
#   message('@0@ exists. Removing...'.format(abs_extract_dir))
#   run_command(python, '-c', rmdir_script, abs_extract_dir, check: true)
# endif

## EXTRACT SLANG
#-------------------------------------------------

message('extracting slang to @0@'.format(abs_extract_dir))
extract_script = '''
import sys, tarfile, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
with tarfile.open(src, 'r:gz') as z:
    z.extractall(dst)
'''
run_command(python, '-c', extract_script, slang_download_path, abs_extract_dir, check: true)

# if get_option('buildtype').startswith('debug')
#   run_command(python, '-c', extract_script, slang_debug_info_download_path, abs_extract_dir, check: true)
# endif

## SETUP DEPENDENCY
#-------------------------------------------------

slang_inc = include_directories(extract_dir / 'include')

cc = meson.get_compiler('cpp')
lib_search_path = [
  abs_extract_dir / 'lib',
  abs_extract_dir / 'bin'
]

install_subdir('bin', install_dir: get_option('bindir'), strip_directory: true)
install_subdir('lib', install_dir: get_option('libdir'), strip_directory: true)

deps = [
  cc.find_library('gfx', dirs: lib_search_path),
  cc.find_library('slang', dirs: lib_search_path),
  cc.find_library('slang-glslang', dirs: lib_search_path, required: false),
  cc.find_library('slang-glsl-module', dirs: lib_search_path, required: false),
  cc.find_library('slang-llvm', dirs: lib_search_path, required: false), # does not exist in the aarch64 package
  cc.find_library('slang-rt', dirs: lib_search_path),
]

exe_search_path = abs_extract_dir / 'bin'
slangc = find_program('slangc', dirs: exe_search_path)
slangd = find_program('slangd', dirs: exe_search_path)
slangi = find_program('slangi', dirs: exe_search_path)

slang_dep = declare_dependency(
  include_directories: slang_inc,
  dependencies: deps,
)
