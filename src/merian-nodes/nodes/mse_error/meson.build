shaders = [
    'mse_image_to_buffer.comp',
    'mse_reduce_buffer.comp',
]

foreach s : shaders
  merian_nodes_src += shader_generator.process(s)
endforeach

merian_nodes_src += files('mse_error.cpp')
