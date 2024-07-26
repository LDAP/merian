shaders = [
    'fullscreen_triangle.vert',
]

foreach s : shaders
  # Header
  merian_src += custom_target(
    s + '.spv.h',
    output: s + '.spv.h',
    command: spv_compile_script + ['--header', s, '@OUTPUT@']
  )

  # Implementation
  merian_src += custom_target(
    s + '.spv.c',
    input: s,
    output: s + '.spv.c',
    command: spv_compile_script + ['@INPUT@', '@OUTPUT@'] + glslc_args,
    depend_files: []
  )
endforeach


merian_src += files(
    'shader_module.cpp',
)
