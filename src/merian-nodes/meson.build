merian_nodes_src = []

subdir('connectors')
subdir('graph')
subdir('nodes')

merian_nodes_lib = static_library(
    'merian-nodes',
    merian_nodes_src,
    dependencies: [
        fmt,
        glfw,
        glm,
        nlohmann_json,
        slang,
        spdlog,
        spirv_reflect,
        stb,
        vma,
        vulkan,
    ],
    include_directories: inc_dirs,
    link_with: merian_lib,
    install : true
)
