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

merian_scene_deps = [
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
]

if is_msvc
    merian_scene_objs = static_library('merian-scene-objs', merian_scene_src,
        dependencies: merian_scene_deps, include_directories: inc_dirs)
    merian_scene_def = custom_target('merian-scene-def',
        input: merian_scene_objs, output: 'merian-scene.def',
        command: [py, gen_def_py, '@OUTPUT@', '@INPUT@'])
    merian_scene_lib = shared_library('merian-scene',
        objects: merian_scene_objs.extract_all_objects(recursive: false),
        vs_module_defs: merian_scene_def, dependencies: merian_scene_deps,
        link_with: merian_lib, install: true)
else
    merian_scene_lib = shared_library('merian-scene', merian_scene_src,
        dependencies: merian_scene_deps, include_directories: inc_dirs,
        link_with: merian_lib, install: true)
endif
