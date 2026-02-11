shaders = [
    'image_to_buffer.comp',
    'reduce_buffer.comp',
]

foreach s : shaders
  merian_nodes_src += shader_generator.process(s, extra_args: ['--target_env', 'vulkan1.3'])
endforeach

merian_nodes_src += files('mean.cpp')
