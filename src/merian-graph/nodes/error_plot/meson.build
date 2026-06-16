shaders = [
    'error_to_buffer.comp',
    'error_reduce_buffer.comp',
]

foreach s : shaders
  merian_graph_src += shader_generator.process(s, extra_args: ['--target_env', 'vulkan1.3'])
endforeach

merian_graph_src += files('error_plot.cpp')
