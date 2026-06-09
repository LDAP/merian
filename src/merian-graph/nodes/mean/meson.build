shaders = [
    'image_to_buffer.comp',
    'reduce_buffer.comp',
]

foreach s : shaders
  merian_graph_src += shader_generator.process(s, extra_args: ['--target_env', 'vulkan1.3'])
endforeach

merian_graph_src += files('mean.cpp')
