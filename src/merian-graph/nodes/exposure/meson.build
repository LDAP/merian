shaders = [
    'exposure.slang',
    'histogram.slang',
    'luminance.slang',
]

foreach s : shaders
  merian_graph_src += shader_generator.process(s)
endforeach

merian_graph_src += files('exposure.cpp')
