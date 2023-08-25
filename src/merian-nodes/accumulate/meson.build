shaders = [
    'accumulate.comp',
]
included_files = files(
    shader_include_dir + '/common/colors_yuv.glsl',
    shader_include_dir + '/common/reprojection.glsl',
    shader_include_dir + '/common/normal_encode.glsl',
    shader_include_dir + '/common/gbuffer.glsl.h',
)
foreach s : shaders
  merian_nodes_src += custom_target(
    s + '.spv.h',
    input: s,
    output: s + '.spv.h',
    # -x save binary output as text-based 32-bit hexadecimal numbers, allows #include as header
    command: [glslc, '-V', '-I' + dir_source_root + '/src', '-I' + shader_include_dir, '--target-env', 'vulkan1.3', '-x', '-o', '@OUTPUT@', '@INPUT@'],
    depend_files: included_files
  )
endforeach

merian_nodes_src += files('accumulate.cpp')
