project('slang',
        'cpp',
        version: '2025.14.3')

fs = import('fs')
python = import('python').find_installation()

slang_zip_format = 'slang-@0@-@1@-@2@.zip'
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

slang_zip = slang_zip_format.format(meson.project_version(), system, host_machine.cpu_family())
slang_url = slang_url_format.format(meson.project_version()) + slang_zip

download_path = meson.current_build_dir() / slang_zip
extract_dir = meson.current_source_dir()

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
if not fs.exists(download_path)
  message('downloading slang from @0@'.format(slang_url))
  run_command(python, '-c', download_script, slang_url, download_path, check: true)
else
  message('skipping download and using @0@'.format(download_path))
endif

message('extracting slang to @0@'.format(extract_dir))
extract_script = '''
import sys, zipfile, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(src, 'r') as z:
    z.extractall(dst)
'''
run_command(python, '-c', extract_script, download_path, extract_dir, check: true)

# Step 3: declare dependency
slang_inc = include_directories('include')

cc = meson.get_compiler('cpp')
search_path = meson.current_source_dir() / 'lib'

deps = [
  cc.find_library('gfx', dirs: search_path),
  cc.find_library('slang', dirs: search_path),
  cc.find_library('slang-glslang', dirs: search_path),
  cc.find_library('slang-glsl-module', dirs: search_path),
  cc.find_library('slang-llvm', dirs: search_path),
  cc.find_library('slang-rt', dirs: search_path),
]

slang_dep = declare_dependency(
  include_directories: slang_inc,
  dependencies: deps
)
