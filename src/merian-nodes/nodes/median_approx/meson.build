shaders = [
    'median_histogram.comp',
    'median_reduce.comp',
]
included_files = files(
    'layout.glsl'
)

foreach s : shaders
  merian_nodes_src += shader_generator.process(s)
endforeach

merian_nodes_src += files('median.cpp')
