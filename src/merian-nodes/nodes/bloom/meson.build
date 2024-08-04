shaders = [
    'bloom_separate.comp',
    'bloom_composite.comp',
]
# included_files = files(
#     'layout.glsl',
#     shader_include_dir + '/common/colors_yuv.glsl',
# )

foreach s : shaders
  merian_nodes_src += shader_generator.process(s)
  # # Header
  # merian_nodes_src += custom_target(
  #   s + '.spv.h',
  #   output: s + '.spv.h',
  #   command: spv_compile_script + ['--header', s, '@OUTPUT@']
  # )

  # # Implementation
  # merian_nodes_src += custom_target(
  #   s + '.spv.c',
  #   input: s,
  #   output: s + '.spv.c',
  #   command: spv_compile_script + ['@INPUT@', '@OUTPUT@'] + glslc_args,
  #   depend_files: included_files
  # )
endforeach

merian_nodes_src += files('bloom.cpp')
