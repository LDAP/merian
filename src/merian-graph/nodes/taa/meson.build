shaders = [
    'taa.slang',      
]

foreach s : shaders
  merian_graph_src += shader_generator.process(s)
endforeach

merian_graph_src += files('taa.cpp')
