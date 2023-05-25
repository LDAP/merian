project(
    'merian',
    'cpp',
    version : '0.0.1',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20'
    ]
)

# Debug configuration
if get_option('buildtype').startswith('debug')
  add_project_arguments('-DDEBUG', language : 'cpp')
  add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
endif

commit = run_command('git', 'rev-parse', 'HEAD', check: true)
version_split = meson.project_version().split('.')
add_project_arguments('-DMERIAN_VERSION="@0@ (@1@)"'.format(meson.project_version(), commit.stdout().strip()), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_MAJOR=@0@'.format(version_split[0]), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_MINOR=@0@'.format(version_split[1]), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_PATCH=@0@'.format(version_split[2]), language : 'cpp')
add_project_arguments('-DMERIAN_PROJECT_NAME="@0@"'.format(meson.project_name()), language : 'cpp')

# Dependencies
fmt = dependency('fmt', version : ['>=8.1.1'], fallback : ['fmt', 'fmt_dep'])
spdlog = dependency('spdlog', version : ['>=1.10.0'], fallback : ['spdlog', 'spdlog_dep'], default_options : ['external_fmt=enabled'])
vulkan = dependency('vulkan', version: ['>=1.3.0'])
glfw = dependency('glfw3', version: ['>=3.3.0'])

# Shaders
glslc = find_program('glslangValidator')
shaders = files(

)
shader_targets = []
foreach s : shaders
    shaders += custom_target(
        'shader @0@'.format(s),
        command : [glslc, '-Ires/shader', '-V', '@INPUT@', '-o',  '@OUTPUT@'],
        input : s, 
        output : '@BASENAME@.spv',
        install_dir: 'assets/shader',
        install: true
    )
endforeach

# Main targets
# Keep sorted
src_files = files(
    'src/utils/stopwatch.cpp',
    'src/vk/context.cpp',
    'src/vk/extension/extension.cpp',
    'src/vk/extension/extension_stopwatch.cpp',
    'src/vk/extension/extension_vk_debug_utils.cpp',
    'src/vk/extension/extension_vk_glfw.cpp',
)
inc_dirs = [
    'include'
]

lib = library(
    'merian',
    src_files,
    dependencies: [
        fmt,
        glfw,
        spdlog,
        vulkan,
    ],
    include_directories: inc_dirs,
    install : true
)

merian_dep = declare_dependency(
    link_with : lib,
    include_directories : inc_dirs
)
