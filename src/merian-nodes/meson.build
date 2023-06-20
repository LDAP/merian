merian_nodes_src = []

subdir('accumulate')
subdir('shadertoy')

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
