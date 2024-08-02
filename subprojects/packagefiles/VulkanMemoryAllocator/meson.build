project(
  'VulkanMemoryAllocator',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().substring(1),
  meson_version: '>=0.64.0',
)

vulkan = dependency('vulkan')
include_dirs = include_directories('include')

implementation = import('fs').copyfile('include/vk_mem_alloc.h', 'vk_mem_alloc.cpp')

vma = library(
  'vma',
  [implementation],
  dependencies: [vulkan],
  cpp_args: ['-DVMA_IMPLEMENTATION'],
  include_directories: include_dirs,
  install: true,
)

VulkanMemoryAllocator_dep = declare_dependency(
  include_directories: include_dirs,
  link_with: vma,
)

VulkanMemoryAllocator_headers_dep = declare_dependency(
  include_directories: include_dirs,
)
