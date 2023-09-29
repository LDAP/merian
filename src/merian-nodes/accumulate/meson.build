shaders = [
    'accumulate.comp',
    'calculate_percentiles.comp',
]
included_files = files(
    shader_include_dir + '/common/colors_yuv.glsl',
    shader_include_dir + '/common/interpolate.glsl',
    'graph_layout.glsl',
)

foreach s : shaders
  # Header
  merian_nodes_src += custom_target(
    s + '.spv.h',
    output: s + '.spv.h',
    command: spv_compile_script + ['--header', s, '@OUTPUT@']
  )

  # Implementation
  merian_nodes_src += custom_target(
    s + '.spv.c',
    input: s,
    output: s + '.spv.c',
    command: spv_compile_script + ['@INPUT@', '@OUTPUT@'] + glslc_args,
    depend_files: included_files
  )
endforeach

merian_nodes_src += files('accumulate.cpp')
