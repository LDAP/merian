shaders = [
    'accumulate.slang',
    'calculate_percentiles.slang',
]

foreach s : shaders
  merian_nodes_src += shader_generator.process(s)
endforeach

merian_nodes_src += files('accumulate.cpp')
