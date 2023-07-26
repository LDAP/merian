shaders = [
    'histogam.comp',      
]
included_files = files(
)
foreach s : shaders
  merian_nodes_src += custom_target(
    s + '.spv.h',
    input: s,
    output: s + '.spv.h',
    # -x save binary output as text-based 32-bit hexadecimal numbers, allows #include as header
    command: [glslc, '-V', '-I' + shader_include_dir, '--target-env', 'vulkan1.3', '-x', '-o', '@OUTPUT@', '@INPUT@'],
    depend_files: included_files
  )
endforeach

# merian_nodes_src += files('taa.cpp')
