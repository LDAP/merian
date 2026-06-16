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

if ufbx.found()
    merian_shaders_src += 'scene/fbx_scene.cpp'
endif

merian_shaders_deps = [
    fmt,
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
]

if is_msvc
    merian_shaders_objs = static_library('merian-shaders-objs', merian_shaders_src,
        dependencies: merian_shaders_deps, include_directories: inc_dirs)
    merian_shaders_def = custom_target('merian-shaders-def',
        input: merian_shaders_objs, output: 'merian-shaders.def',
        command: [py, gen_def_py, '@OUTPUT@', '@INPUT@'])
    merian_shaders_lib = shared_library('merian-shaders',
        objects: merian_shaders_objs.extract_all_objects(recursive: false),
        vs_module_defs: merian_shaders_def, dependencies: merian_shaders_deps,
        link_with: merian_lib, install: true)
else
    merian_shaders_lib = shared_library('merian-shaders', merian_shaders_src,
        dependencies: merian_shaders_deps, include_directories: inc_dirs,
        link_with: merian_lib, install: true)
endif
