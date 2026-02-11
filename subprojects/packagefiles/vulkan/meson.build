project(
  'vulkan',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().substring(1),
  meson_version: '>=0.64.0',
)

# Vulkan Headers

include_dirs = include_directories('include')

vulkanheaders_dep = declare_dependency(
  include_directories: include_dirs,
)

# Vulkan Loader

ccx = meson.get_compiler('cpp')
fs = import('fs')
py = import('python').find_installation('python3')

vulkan_sdk = run_command(
    py, '-c', 'import os; print(os.environ.get("VULKAN_SDK", os.environ.get("VK_SDK_PATH", "")))',
    check: true
).stdout().strip()
search_paths = []
lib_name = 'vulkan'

if vulkan_sdk != ''
    if host_machine.system() == 'windows'
        lib_name = 'vulkan-1'
        inc_dir = 'Include'
        host_cpu = host_machine.cpu_family()

        if host_cpu == 'x86_64'
            lib_dir = 'Lib'
        elif host_cpu == 'aarch64'
            lib_dir = 'Lib-ARM64'
        elif host_cpu == 'x86'
            lib_dir = 'Lib32'
        else
            error('Architecture @0@'.format(host_cpu))
        endif
    else
        lib_dir = 'lib'
        inc_dir = 'include'
    endif
    lib_path = vulkan_sdk / lib_dir

    if not fs.is_dir(lib_path)
        error('VULKAN_SDK points to an invalid directory: ' + vulkan_sdk)
    endif
    search_paths += lib_path
endif

vulkan_loader = ccx.find_library(lib_name, dirs: search_paths, required: true)

vulkan_dep = declare_dependency(
    dependencies: [vulkanheaders_dep, vulkan_loader],
)

