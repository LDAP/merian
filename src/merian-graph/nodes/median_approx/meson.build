shaders = [
    'median_histogram.comp',
    'median_reduce.comp',
]
included_files = files(
    'layout.glsl'
)

foreach s : shaders
  merian_graph_src += shader_generator.process(s)
endforeach

merian_graph_src += files('median.cpp')
