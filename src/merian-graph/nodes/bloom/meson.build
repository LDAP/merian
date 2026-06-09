shaders = [
    'bloom_separate.slang',
    'bloom_composite.slang',
]
# included_files = files(
#     'layout.glsl',
#     shader_include_dir + '/common/colors_yuv.glsl',
# )

foreach s : shaders
  merian_graph_src += shader_generator.process(s)
  # # Header
  # merian_graph_src += custom_target(
  #   s + '.spv.h',
  #   output: s + '.spv.h',
  #   command: spv_compile_script + ['--header', s, '@OUTPUT@']
  # )

  # # Implementation
  # merian_graph_src += custom_target(
  #   s + '.spv.c',
  #   input: s,
  #   output: s + '.spv.c',
  #   command: spv_compile_script + ['@INPUT@', '@OUTPUT@'] + glslc_args,
  #   depend_files: included_files
  # )
endforeach

merian_graph_src += files('bloom.cpp')
