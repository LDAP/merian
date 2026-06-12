merian_scene_src = [
    'scene.cpp',
    'material_system.cpp',
    'texture_manager.cpp',
]

shaders = [
    'transform_vertex.slang',
    'transform_prev_vertex.slang',
]
foreach s : shaders
    merian_scene_src += shader_generator.process(s)
endforeach

if tgltf.found()
    merian_scene_src += 'gltf_scene.cpp'
endif

if ufbx.found()
    merian_scene_src += 'fbx_scene.cpp'
endif

merian_scene_lib = shared_library(
    'merian-scene',
    merian_scene_src,
    dependencies: [
        fmt,
        glm,
        nlohmann_json,
        slang,
        spdlog,
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
