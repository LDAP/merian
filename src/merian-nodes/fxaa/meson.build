shaders = [
    'fxaa.comp',
]
included_files = files(
    'fxaa.h'
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

merian_nodes_src += files('fxaa.cpp')
