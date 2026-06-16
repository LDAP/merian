merian_graph_src = []

subdir('connectors')
subdir('graph')
subdir('nodes')

merian_graph_deps = [
    fmt,
    glfw,
    glm,
    imgui,
    nlohmann_json,
    slang,
    spdlog,
    spirv_reflect,
    stb,
    tgltf,
    ufbx,
    vma,
    vulkan,
]

if is_msvc
    merian_graph_objs = static_library('merian-graph-objs', merian_graph_src,
        dependencies: merian_graph_deps, include_directories: inc_dirs)
    merian_graph_def = custom_target('merian-graph-def',
        input: merian_graph_objs, output: 'merian-graph.def',
        command: [py, gen_def_py, '@OUTPUT@', '@INPUT@'])
    merian_graph_lib = shared_library('merian-graph',
        objects: merian_graph_objs.extract_all_objects(recursive: false),
        vs_module_defs: merian_graph_def, dependencies: merian_graph_deps,
        link_with: [merian_lib, merian_shaders_lib], install: true)
else
    merian_graph_lib = shared_library('merian-graph', merian_graph_src,
        dependencies: merian_graph_deps, include_directories: inc_dirs,
        link_with: [merian_lib, merian_shaders_lib], install: true)
endif
