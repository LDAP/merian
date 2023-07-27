shader_include_dir = dir_source_root + '/src/merian-nodes'
merian_nodes_src = []

subdir('ab_compare')
subdir('accumulate')
subdir('color_output')
subdir('compute_node')
subdir('exposure')
subdir('image')
subdir('image_write')
subdir('shadertoy')
subdir('shadertoy_spheres')
subdir('svgf')
subdir('taa')
subdir('tonemap')
subdir('vkdt_filmcurv')

merian_nodes_lib = static_library(
    'merian-nodes',
    merian_nodes_src,
    dependencies: [
        fmt,
        glm,
        spdlog,
        vulkan,
    ],
    include_directories: inc_dirs,
    link_with: merian_lib,
    install : true
)
