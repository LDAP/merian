project(
    'merian',
    ['cpp', 'c'],
    version : '1.0.0',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
    ]
)

cmake = import('cmake')

ccx = meson.get_compiler('cpp')
glslc = find_program('glslangValidator')

dir_source_root = meson.project_source_root()

if get_option('loglevel') == 'auto'
    if get_option('buildtype').startswith('debug')
        add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
    endif
else
    add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_' + get_option('loglevel').to_upper(), language : 'cpp')
endif

if get_option('performance_profiling')
  add_project_arguments('-DMERIAN_PROFILER_ENABLE', language : 'cpp')
endif

# Vulkan dynamic loader (VULKAN_HPP_DEFAULT_DISPATCHER initialized in Context)
add_project_arguments('-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1', language : 'cpp')
# Vulkan uses [0, 1] for depth instead of [-1, 1] like OpenGL
add_project_arguments('-DGLM_FORCE_DEPTH_ZERO_TO_ONE', language : 'cpp')

commit = run_command('git', 'rev-parse', 'HEAD', check: true)
version_split = meson.project_version().split('.')
add_project_arguments('-DMERIAN_VERSION="@0@ (@1@)"'.format(meson.project_version(), commit.stdout().strip()), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_MAJOR=@0@'.format(version_split[0]), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_MINOR=@0@'.format(version_split[1]), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_PATCH=@0@'.format(version_split[2]), language : 'cpp')
add_project_arguments('-DMERIAN_PROJECT_NAME="@0@"'.format(meson.project_name()), language : 'cpp')

# Dependencies
fmt = dependency('fmt', version : ['>=10.0.0'], fallback : ['fmt', 'fmt_dep'])
spdlog = dependency('spdlog', version : ['>=1.12.0'], fallback : ['spdlog', 'spdlog_dep'], default_options : ['external_fmt=enabled'])
vulkan = dependency('vulkan', version: ['>=1.3.0'], required: false)
if not vulkan.found()
    # version specifier is broken for vulkan https://github.com/mesonbuild/meson/issues/12072
    vulkan = dependency('vulkan')
else
    
endif
glfw = dependency('glfw3', version: ['>=3.4.0'], fallback: ['glfw', 'glfw_dep'])
glm = dependency('glm', fallback : ['glm', 'glm_dep'], version: ['>=0.9.9.8'])
tol_subp = cmake.subproject('tinyobjloader')
tol = tol_subp.dependency('tinyobjloader')
tgltf = dependency('tinygltf', required: get_option('tinygltf'), fallback: ['tinygltf', 'tinygltf_dep'])
imgui = dependency('imgui', version: ['>=1.90.9'], fallback: ['imgui', 'imgui_dep'])
nlohmann_json = dependency('nlohmann_json', version: ['>=3.11.3'], fallback: ['nlohmann_json', 'nlohmann_json_dep'])
vma = dependency('VulkanMemoryAllocator', version: ['>=3.1.0'], fallback: ['VulkanMemoryAllocator', 'VulkanMemoryAllocator_dep'])
stb = dependency('stb', fallback: ['stb', 'stb_dep'], default_options : ['stb_libraries=[\'stb_image\', \'stb_image_write\']'])
sdl2 = dependency('sdl2', required: false)
if not sdl2.found()
    # Workaround because SDL2 ships Vulkan which conflicts
    # This is basically the same declare_dependency they have without "deps"
    sdl2_subp = subproject('sdl2')
    sdl2 = declare_dependency(
        link_with : sdl2_subp.get_variable('sdl2'),
        include_directories : sdl2_subp.get_variable('core_inc'),
        compile_args : sdl2_subp.get_variable('c_args')
    )
endif
shaderc = dependency('shaderc', version: ['>=2023.8.1'], required: false)
if not shaderc.found() and get_option('shaderc').enabled()
    shaderc_options = cmake.subproject_options()
    shaderc_options.add_cmake_defines({'SHADERC_SKIP_TESTS': true})
    shaderc_options.add_cmake_defines({'SHADERC_SKIP_GLSLC': true})
    shaderc_options.add_cmake_defines({'SHADERC_SKIP_EXAMPLES': true})
    shaderc_options.add_cmake_defines({'SHADERC_ENABLE_SHARED_CRT': true})
    shaderc_subp = cmake.subproject('shaderc', options: shaderc_options)
    # ideally we would use shaderc_subp.dependency('shaderc_combined') but that does not work
    # therefore we combine all dependencies into one (shaderc build is pretty messed up)
    shaderc = declare_dependency(
        dependencies: [
            shaderc_subp.dependency('shaderc'),
            shaderc_subp.dependency('glslang'),
            shaderc_subp.dependency('MachineIndependent'),
            shaderc_subp.dependency('GenericCodeGen'),
            shaderc_subp.dependency('shaderc_util'),
            shaderc_subp.dependency('SPIRV'),
            shaderc_subp.dependency('SPIRV-Tools-static'),
            shaderc_subp.dependency('SPIRV-Tools-opt'),
        ],
    )
endif

if get_option('shaderc').enabled()
    add_project_arguments('-DMERIAN_ENABLE_SHADERC', language: 'cpp')
endif

inc_dirs = [
    include_directories('./src'),
]

shader_include_dir = dir_source_root + '/src/merian-nodes'
spv_compile_script = [dir_source_root + '/scripts/compile_shader.py']
glslc_args = ['-I' + dir_source_root + '/src', '-I' + shader_include_dir,'--target-env', 'vulkan1.3']

subdir('src/merian')
subdir('src/merian-nodes')

merian_dep = declare_dependency(
    link_with : [merian_lib, merian_nodes_lib],
    include_directories : inc_dirs,
    dependencies : [
        fmt,
        glfw,
        glm,
        imgui,
        nlohmann_json,
        spdlog,
        tol,
        vma,
        vulkan,
    ]
)
