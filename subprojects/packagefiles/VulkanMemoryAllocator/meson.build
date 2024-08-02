project(
  'VulkanMemoryAllocator',
  'cpp',
  license: 'MIT',
  version: run_command(['git', 'describe', '--exact-match', '--tags']).stdout().substring(1),
  meson_version: '>=0.56.0',
)

vulkan = dependency('vulkan')
include_dirs = include_directories('include')


python3 = import('python').find_installation('python3')

implementation = custom_target('vk_mem_alloc',
  output: 'vk_mem_alloc.cpp',
  command: [python3, '-c', '''
with open('@OUTPUT@', 'w') as f:
    f.write(r"#include <vk_mem_alloc.h>")
''']
)

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
