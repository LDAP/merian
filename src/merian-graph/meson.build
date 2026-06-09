merian_graph_src = []

subdir('connectors')
subdir('graph')
subdir('nodes')

merian_graph_lib = static_library(
    'merian-graph',
    merian_graph_src,
    dependencies: [
        fmt,
        glfw,
        glm,
        nlohmann_json,
        slang,
        spdlog,
        spirv_reflect,
        stb,
        tgltf,
        ufbx,
        vma,
        vulkan,
    ],
    include_directories: inc_dirs,
    link_with: merian_lib,
    install : true
)
