project(
    'merian',
    ['cpp', 'c'],
    version : run_command(['git', 'describe', '--tags', '--abbrev=0'], check: true).stdout().strip(),
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_lto=true',
        'b_ndebug=if-release',
    ],
    meson_version: '>=1.5.0',
)

# attempt to stick to Vulkan SDK versions (vulkan-sdk-X.X.XXX.X tags and update the version in the wrap file).
vulkan_target_api_version = '1.4.350'
# update together with SPIRV-Reflect, SPIRV-headers and glslang wrap
spirv_headers_version = 'vulkan-sdk-1.4.350.1'

cmake = import('cmake')
py = import('python').find_installation('python3')
fs = import('fs')

ccx = meson.get_compiler('cpp')

# export all objects on windows
is_msvc = ccx.get_argument_syntax() == 'msvc'
gen_def_py = files('scripts/gen_def.py')

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
    # Guides the default optimization of the runtime shader compiler 
    '-DMERIAN_BUILD_OPTIMIZATION=@0@'.format(get_option('optimization')),

    # Vulkan uses [0, 1] for depth instead of [-1, 1] like OpenGL
    '-DGLM_FORCE_DEPTH_ZERO_TO_ONE',
    '-DGLM_FORCE_SWIZZLE=1'
]

# the dynamic dispatcher storage lives in merian-core's dll; consumers import it
if is_msvc
    global_args += ['-DVULKAN_HPP_STORAGE_SHARED']
endif

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
vulkan = dependency('vulkan', version: ['>=' + vulkan_target_api_version])

spirv_headers_subp = cmake.subproject('SPIRV-Headers')
spirv_headers = spirv_headers_subp.dependency('SPIRV-Headers')
meson.override_dependency('SPIRV-Headers', spirv_headers)

fmt = dependency('fmt', version : ['>=12.1.0'], fallback : ['fmt', 'fmt_dep'])
spdlog = dependency('spdlog', version : ['>=1.17.0'], fallback : ['spdlog', 'spdlog_dep'], default_options : ['external_fmt=enabled'])
spirv_reflect = dependency('spirv-reflect', fallback: ['spirv-reflect', 'spirv_reflect_dep'])
glfw = dependency('glfw3', version: ['>=3.4.0'], required: get_option('glfw'), fallback: ['glfw', 'glfw_dep'])
if get_option('glfw').enable_auto_if(glfw.found()).enabled()
    global_args += ['-DMERIAN_GLFW_ENABLED']
endif
subprocess = dependency('subprocess', version: ['>=0.4.0'], required: false)
if not subprocess.found()
    # Built from the CMake wrap; force PIC so it can be linked into the shared merian libraries.
    subprocess_options = cmake.subproject_options()
    subprocess_options.add_cmake_defines({'CMAKE_POSITION_INDEPENDENT_CODE': true})
    subprocess_subp = cmake.subproject('subprocess', options: subprocess_options)
    subprocess = subprocess_subp.dependency('subprocess')
endif
glm = dependency('glm', fallback : ['glm', 'glm_dep'], version: ['>=1.0.3'])
tol = dependency('tinyobjloader', required: false)
if not tol.found()
    # Built from the CMake wrap; force PIC so it can be linked into the shared merian libraries.
    tol_options = cmake.subproject_options()
    tol_options.add_cmake_defines({'CMAKE_POSITION_INDEPENDENT_CODE': true})
    tol_subp = cmake.subproject('tinyobjloader', options: tol_options)
    tol = tol_subp.dependency('tinyobjloader')
endif
tgltf = dependency('tinygltf', required: get_option('tinygltf'), fallback: ['tinygltf', 'tinygltf_dep'])
if get_option('tinygltf').enable_auto_if(tgltf.found()).enabled()
    global_args += ['-DMERIAN_TINYGLTF_ENABLED']
endif
ufbx = dependency('ufbx', required: get_option('ufbx'), fallback: ['ufbx', 'ufbx_dep'])
if get_option('ufbx').enable_auto_if(ufbx.found()).enabled()
    global_args += ['-DMERIAN_UFBX_ENABLED']
endif
imgui = dependency('imgui', version: ['>=1.92.6'], fallback: ['imgui', 'imgui_dep'], static: true)
nlohmann_json = dependency('nlohmann_json', version: ['>=3.11.3'], fallback: ['nlohmann_json', 'nlohmann_json_dep'])
vma = dependency('VulkanMemoryAllocator', version: ['>=3.3.0'], fallback: ['VulkanMemoryAllocator', 'VulkanMemoryAllocator_dep'], default_options : ['static_functions=false', 'dynamic_functions=true'])
stb = dependency('stb', fallback: ['stb', 'stb_dep'], default_options : ['stb_libraries=[\'stb_image\', \'stb_image_write\']'])
tinybvh = dependency('tinybvh', required: false)
if not tinybvh.found() and get_option('tinybvh').enabled()
    tinybvh = dependency('tinybvh', fallback: ['tinybvh', 'tinybvh_dep'])
endif
sdl3 = dependency('sdl3', version: ['>=3.4.0'], required: get_option('sdl'), fallback: ['sdl3', 'sdl3_dep'])
if get_option('sdl').enable_auto_if(sdl3.found()).enabled()
    global_args += ['-DMERIAN_SDL_ENABLED']
endif
glslc = find_program('glslangValidator', required: false)
glslang = dependency('glslang', version: ['>=16.2.0'], required: false, method: 'cmake',
    modules: ['glslang::glslang', 'glslang::glslang-default-resource-limits', 'glslang::SPIRV'])
if glslang.found()
    # glslang::SPIRV depends on SPIRV-Tools but there may be no cmake config for it,
    # so meson's cmake integration doesn't resolve the transitive dependency automatically.
    # see https://github.com/mesonbuild/meson/issues/8232
    glslang = declare_dependency(dependencies: [glslang,
        ccx.find_library('SPIRV-Tools', required: false),
        ccx.find_library('SPIRV-Tools-opt', required: false),
        ccx.find_library('SPIRV-Tools-link', required: false),
    ])
endif
shader_generator_deps = []
if not glslc.found() or (not glslang.found() and get_option('glslang').enabled())
    glslang_options = cmake.subproject_options()
    glslang_options.add_cmake_defines({'GLSLANG_TESTS': false})
    glslang_options.add_cmake_defines({'GLSLANG_ENABLE_INSTALL': false})
    glslang_options.add_cmake_defines({'ENABLE_SPIRV': true})
    glslang_options.set_install(false)
    # installs as glslang_standalone
    # glslang_options.set_install(true, target: 'glslang-standalone')
    glslang_subp = cmake.subproject('glslang', options: glslang_options)
    glslc = glslang_subp.target('glslang-standalone')
    glslang = declare_dependency(dependencies: [
        glslang_subp.dependency('glslang'),
        glslang_subp.dependency('glslang-default-resource-limits'),
        glslang_subp.dependency('SPIRV'),
    ])
    shader_generator_deps = [glslc]
endif
if get_option('glslang').enable_auto_if(glslang.found()).enabled()
    global_args += ['-DMERIAN_GLSLANG_ENABLED']
endif
glslc_path = glslc.full_path()

slang = dependency('slang', version: ['>=2026.5.2'], fallback: ['slang', 'slang_dep'])
slangc = find_program('slangc', required: false)
if not slangc.found()
    slangc = subproject('slang').get_variable('slangc')
endif

generator_scripts = [
    'scripts/generate_enum_string_map.py',
    'scripts/generate_extensions.py',
    'scripts/generate_features.py',
    'scripts/generate_properties.py',
    'scripts/generate_spirv.py',
]
message('Generating Vulkan code generators for version v@0@...'.format(vulkan_target_api_version))
foreach script : generator_scripts
    message(run_command(
        py, meson.current_source_dir() / script,
        env: {
            'VULKAN_SPEC_VERSION': 'v' + vulkan_target_api_version,
            'SPIRV_HEADERS_VERSION': spirv_headers_version,
            'MERIAN_BUILD_DIR': meson.global_build_root(),
        },
        check: true
    ).stdout())
endforeach


project_root = meson.current_source_dir()
inc_dirs = [
    include_directories('./include'),
]
spv_compile_script = [
    meson.project_source_root() + '/scripts/compile_shader.py',
    '--glslc_path', glslc_path,
    '--slangc_path', slangc.full_path(),
    '--optimization', get_option('optimization')
]
spv_compiler_args = ['-I' + meson.project_source_root()]
spv_compiler_args += ['-I' + join_paths(project_root, 'src')]
spv_compiler_args += ['-I' + join_paths(project_root, 'include')]
spv_compiler_args += ['--depfile', '@DEPFILE@']
if get_option('buildtype').startswith('debug')
    spv_compiler_args += ['-g']
endif

shader_generator = generator(
    py,
    output: ['@PLAINNAME@.spv.h', '@PLAINNAME@.spv.c'],
    depfile: '@PLAINNAME@.d',
    arguments: spv_compile_script + ['@INPUT@', '@OUTPUT0@', '@OUTPUT1@'] + spv_compiler_args + ['@EXTRA_ARGS@'],
    depends: shader_generator_deps
)


add_project_arguments(global_args, language: 'cpp')

subdir('src/merian')
subdir('src/merian-shaders')
subdir('src/merian-graph')

merian_dep = declare_dependency(
    compile_args: global_args,
    link_with : [merian_lib, merian_shaders_lib, merian_graph_lib],
    include_directories : inc_dirs,
    dependencies : [
        fmt,
        glfw.found() ? glfw : [],
        glm,
        glslang,
        imgui,
        nlohmann_json,
        slang,
        spdlog,
        spirv_reflect,
        tinybvh,
        tgltf,
        tol,
        vma,
        vulkan,
    ]
)

merian_graph_run = executable(
    'merian-graph-run',
    'src/merian-graph-run/main.cpp',
    dependencies: merian_dep,
    export_dynamic: true,
    install: true,
)

# Let external plugin projects consume merian without vendoring. Use build/meson-uninstalled for
# building against the build tree directly.
pkgconfig = import('pkgconfig')
pkgconfig.generate(
    libraries: [merian_lib, merian_shaders_lib, merian_graph_lib],
    name: 'merian',
    filebase: 'merian',
    description: 'Merian Vulkan development framework',
    extra_cflags: global_args,
    requires: [vulkan, fmt, spdlog, nlohmann_json],
)
meson.override_dependency('merian', merian_dep)

# Build any plugin cloned into plugins/.
plugin_names = run_command(find_program('python3'), '-c', '''
import os, sys
root, links = sys.argv[1], sys.argv[2]
out = []
if os.path.isdir(root):
    for name in sorted(os.listdir(root)):
        if not os.path.isfile(os.path.join(root, name, "meson.build")):
            continue
        link, target = os.path.join(links, name), os.path.join("..", "plugins", name)
        if not (os.path.islink(link) and os.readlink(link) == target):
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(target, link)
        out.append(name)
print("\\n".join(out))
''', meson.project_source_root() / 'plugins', meson.project_source_root() / 'subprojects',
    check: true).stdout().strip()

foreach name : plugin_names.split('\n')
    if name != ''
        subproject(name, required: false)
    endif
endforeach

install_subdir('include', install_dir: get_option('includedir'), strip_directory: true)
install_subdir('examples', install_dir: join_paths(get_option('datadir'), 'merian'))
install_data('scripts/compile_shader.py',
             install_dir: join_paths(get_option('datadir'), 'merian', 'scripts'))

if get_option('tests')
    subdir('tests')
endif
