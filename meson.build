project(
    'merian',
    ['cpp', 'c'],
    version : run_command(['git', 'describe', '--tags', '--abbrev=0']).stdout().strip(),
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
    ],
    meson_version: '>=1.5.0',
)

cmake = import('cmake')

ccx = meson.get_compiler('cpp')

if get_option('loglevel') == 'auto'
    if get_option('buildtype').startswith('debug')
        add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
    endif
else
    add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_' + get_option('loglevel').to_upper(), language : 'cpp')
endif

global_args = [
    # Vulkan dynamic loader (VULKAN_HPP_DEFAULT_DISPATCHER initialized in Context)
    '-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1',
    # Vulkan uses [0, 1] for depth instead of [-1, 1] like OpenGL
    '-DGLM_FORCE_DEPTH_ZERO_TO_ONE',
]

if get_option('performance_profiling')
  global_args += ['-DMERIAN_PROFILER_ENABLE']
endif

commit = run_command('git', 'rev-parse', 'HEAD', check: true)
version_split = meson.project_version().split('.')
add_project_arguments('-DMERIAN_VERSION="@0@ (@1@)"'.format(meson.project_version(), commit.stdout().strip()), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_MAJOR=@0@'.format(version_split[0]), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_MINOR=@0@'.format(version_split[1]), language : 'cpp')
add_project_arguments('-DMERIAN_VERSION_PATCH=@0@'.format(version_split[2]), language : 'cpp')
add_project_arguments('-DMERIAN_PROJECT_NAME="@0@"'.format(meson.project_name()), language : 'cpp')

add_project_arguments('-DMERIAN_INSTALL_PREFIX="@0@"'.format(get_option('prefix')), language : 'cpp')
add_project_arguments('-DMERIAN_INCLUDE_DIR_NAME="@0@"'.format(get_option('includedir')), language : 'cpp')
add_project_arguments('-DMERIAN_DATA_DIR_NAME="@0@"'.format(get_option('datadir')), language : 'cpp')
add_project_arguments('-DMERIAN_DEVELOPMENT_INCLUDE_DIR="@0@"'.format(join_paths(meson.current_source_dir(), 'include')), language : 'cpp')

# Dependencies
fmt = dependency('fmt', version : ['>=10.0.0'], fallback : ['fmt', 'fmt_dep'])
spdlog = dependency('spdlog', version : ['>=1.12.0'], fallback : ['spdlog', 'spdlog_dep'], default_options : ['external_fmt=enabled'])
vulkan = dependency('vulkan', version: ['>=1.3.0'], required: false)
if not vulkan.found()
    # version specifier is broken for vulkan in older meson versions. https://github.com/mesonbuild/meson/issues/12072
    vulkan = dependency('vulkan')
endif
vulkan = vulkan.partial_dependency(compile_args: true, link_args: true, links: true)
vulkanheaders = subproject('vulkanheaders').get_variable('vulkanheaders_dep')
glfw = dependency('glfw3', version: ['>=3.4.0'], fallback: ['glfw', 'glfw_dep'])
subprocess = dependency('subprocess', version: ['>=0.4.0'], fallback: ['subprocess', 'subprocess_dep'])
glm = dependency('glm', fallback : ['glm', 'glm_dep'], version: ['>=0.9.9.8'])
tol = dependency('tinyobjloader', fallback : ['tinyobjloader', 'tinyobjloader_dep'])
tgltf = dependency('tinygltf', required: get_option('tinygltf'), fallback: ['tinygltf', 'tinygltf_dep'])
imgui = dependency('imgui', version: ['>=1.90.9'], fallback: ['imgui', 'imgui_dep'], static: true)
nlohmann_json = dependency('nlohmann_json', version: ['>=3.11.3'], fallback: ['nlohmann_json', 'nlohmann_json_dep'])
vma = dependency('VulkanMemoryAllocator', version: ['>=3.1.0'], fallback: ['VulkanMemoryAllocator', 'VulkanMemoryAllocator_dep'])
stb = dependency('stb', fallback: ['stb', 'stb_dep'], default_options : ['stb_libraries=[\'stb_image\', \'stb_image_write\']'])
tinybvh = dependency('tinybvh', fallback: ['tinybvh', 'tinybvh_dep'])
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
    shaderc_options.add_cmake_defines({'SHADERC_SKIP_EXAMPLES': true})
    shaderc_options.add_cmake_defines({'SHADERC_ENABLE_SHARED_CRT': true})
    shaderc_subp = cmake.subproject('shaderc', options: shaderc_options)
    # ideally we would use shaderc_subp.dependency('shaderc_combined') but that does not work
    # therefore we combine all dependencies into one.
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
if shaderc.found()
    global_args += ['-DMERIAN_SHADERC_FOUND']
endif

project_root = meson.current_source_dir()
inc_dirs = [
    include_directories('./include'),
]

glslc = find_program('glslangValidator', required: false)
if glslc.found()
    glslc_path = glslc.full_path()
    shader_generator_deps = []
else
    # :( in this case we have to build glslangValidator ourselves
    glslang_options = cmake.subproject_options()
    glslang_options.add_cmake_defines({'GLSLANG_TESTS': false})
    glslang_options.add_cmake_defines({'GLSLANG_ENABLE_INSTALL': false})
    glslang_options.set_install(false)
    # installs as glslang_standalone
    # glslang_options.set_install(true, target: 'glslang-standalone')
    glslang_subp = cmake.subproject('glslang', options: glslang_options)
    glslc_tgt = glslang_subp.target('glslang-standalone')
    glslc_path = glslc_tgt.full_path()
    shader_generator_deps = [glslc_tgt]
endif

spv_compile_script = [meson.project_source_root() + '/scripts/compile_shader.py', '--glslc_path', glslc_path]
glslc_args = ['--target-env', 'vulkan1.3']
glslc_args += ['-I' + meson.project_source_root()]
glslc_args += ['-I' + join_paths(project_root, 'src')]
glslc_args += ['-I' + join_paths(project_root, 'include')]
glslc_args += ['--depfile', '@DEPFILE@']
if get_option('buildtype').startswith('debug')
    glslc_args += ['-g']
endif

py = import('python').find_installation('python3') 
shader_generator = generator(
    py,
    output: ['@PLAINNAME@.spv.h', '@PLAINNAME@.spv.c'],
    depfile: '@PLAINNAME@.d',
    arguments: spv_compile_script + ['@INPUT@', '@OUTPUT0@', '@OUTPUT1@'] + glslc_args + ['@EXTRA_ARGS@'],
    depends: shader_generator_deps
)


add_project_arguments(global_args, language: 'cpp')

subdir('src/merian')
subdir('src/merian-nodes')

merian_dep = declare_dependency(
    compile_args: global_args,
    link_with : [merian_lib, merian_nodes_lib],
    include_directories : inc_dirs,
    dependencies : [
        fmt,
        glfw,
        glm,
        imgui,
        nlohmann_json,
        shaderc,
        spdlog,
        tinybvh,
        tol,
        vma,
        vulkan,
        vulkanheaders,
    ]
)

install_subdir('include', install_dir: get_option('includedir'), strip_directory: true)
