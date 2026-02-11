project(
  'VulkanMemoryAllocator',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().substring(1),
  meson_version: '>=0.64.0',
)

global_args = []

if get_option('static_functions') and get_option('dynamic_functions')
  error('static_functions and dynamic_functions cannot be both true')
endif

global_args += '-DVMA_STATIC_VULKAN_FUNCTIONS=@0@'.format(get_option('static_functions') ? '1' : '0')
global_args += '-DVMA_DYNAMIC_VULKAN_FUNCTIONS=@0@'.format(get_option('dynamic_functions') ? '1' : '0')

vulkan = dependency('vulkan')
include_dirs = include_directories('include')

implementation = import('fs').copyfile('include/vk_mem_alloc.h', 'vk_mem_alloc.cpp')

vma = static_library(
  'vma',
  [implementation],
  dependencies: [vulkan],
  cpp_args: ['-DVMA_IMPLEMENTATION'] + global_args,
  include_directories: include_dirs,
  install: true,
)

VulkanMemoryAllocator_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: vma,
  compile_args: global_args,
)

VulkanMemoryAllocator_headers_dep = declare_dependency(
  include_directories: include_dirs,
  compile_args: global_args,
)
