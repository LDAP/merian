shaders = [
    'vkdt_filmcurv.slang',      
]

foreach s : shaders
  merian_nodes_src += shader_generator.process(s)
endforeach

merian_nodes_src += files('vkdt_filmcurv.cpp')
