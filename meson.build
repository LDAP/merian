project(
    'merian',
    'cpp',
    version : '0.0.1',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
    ]
)

cmake = import('cmake')
glslc = find_program('glslangValidator')

dir_source_root = meson.project_source_root()

# Debug configuration
if get_option('buildtype').startswith('debug')
  add_project_arguments('-DDEBUG', language : 'cpp')
  add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
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
fmt = dependency('fmt', version : ['>=8.1.1'], fallback : ['fmt', 'fmt_dep'])
spdlog = dependency('spdlog', version : ['>=1.10.0'], fallback : ['spdlog', 'spdlog_dep'], default_options : ['external_fmt=enabled'])
vulkan = dependency('vulkan', version: ['>=1.3.0'])
glfw = dependency('glfw3', version: ['>=3.3.0'])
glm = dependency('glm', fallback : ['glm', 'glm_dep'], version: ['>=0.9.9.8'])
tol_subp = cmake.subproject('tinyobjloader')
tol = tol_subp.dependency('tinyobjloader')
tgltf = cmake.subproject('tinygltf').dependency('tinygltf')

inc_dirs = [
    include_directories('./src'),
    cmake.subproject('VulkanMemoryAllocator').include_directories('VulkanMemoryAllocator'),
    tol_subp.include_directories('tinyobjloader')
]

subdir('src/ext')
subdir('src/merian')
subdir('src/merian-nodes')

# Keep sorted

merian_dep = declare_dependency(
    link_with : [ext_lib, merian_lib, merian_nodes_lib],
    include_directories : inc_dirs
)
