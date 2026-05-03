merian_shaders_src = [
    'scene/scene.cpp',
    'shading/materials/material_system.cpp',
    'utils/texture_manager.cpp',
]

shaders = [
    'scene/transform_vertex.slang',
    'scene/transform_prev_vertex.slang',
]
foreach s : shaders
    merian_shaders_src += shader_generator.process(s)
endforeach


if tgltf.found()
    merian_shaders_src += 'scene/gltf_scene.cpp'
endif

merian_shaders_lib = static_library(
    'merian-shaders',
    merian_shaders_src,
    dependencies: [
        fmt,
        glm,
        nlohmann_json,
        slang,
        spdlog,
        stb,
        tgltf,
        vma,
        vulkan,
    ],
    include_directories: inc_dirs,
    link_with: merian_lib,
    install : true
)
