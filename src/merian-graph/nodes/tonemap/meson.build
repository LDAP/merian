shaders = [
    'tonemap.slang',
]

foreach s : shaders
  merian_graph_src += shader_generator.process(s)
endforeach

merian_graph_src += files('tonemap.cpp')
