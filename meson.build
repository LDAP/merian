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
add_project_arguments('-DVERSION="@0@ (@1@)"'.format(meson.project_version(), commit.stdout().strip()), language : 'cpp')
add_project_arguments('-DVERSION_MAJOR=@0@'.format(version_split[0]), language : 'cpp')
add_project_arguments('-DVERSION_MINOR=@0@'.format(version_split[1]), language : 'cpp')
add_project_arguments('-DVERSION_PATCH=@0@'.format(version_split[2]), language : 'cpp')
add_project_arguments('-DPROJECT_NAME="@0@"'.format(meson.project_name()), language : 'cpp')

# Dependencies
fmt = dependency('fmt', version : ['>=8.1.1'], fallback : ['fmt', 'fmt_dep'])
spdlog = dependency('spdlog', version : ['>=1.10.0'], fallback : ['spdlog', 'spdlog_dep'], default_options : ['external_fmt=enabled'])
vulkan = dependency('vulkan')
glfw = dependency('glfw3')

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
    'src/merian.cpp',
    'src/vk/context.cpp',
    'src/vk/extension/extension.cpp',
    'src/vk/extension/extension_debug_utils.cpp',
    'src/vk/extension/extension_glfw.cpp',
)
inc_dirs = [
    'include'
]

exe = executable(
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

test('basic', exe)
