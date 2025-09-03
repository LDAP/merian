merian_src += shader_generator.process('fullscreen_triangle.vert',
                                       preserve_path_from: meson.current_source_dir())

merian_src += files(
    'entry_point.cpp',
    'glsl_shader_compiler.cpp',
    'glsl_shader_compiler_system_glslangValidator.cpp',
    'glsl_shader_compiler_system_glslc.cpp',
    'shader_compiler.cpp',
    'shader_hotreloader.cpp',
    'shader_module.cpp',
    'slang_composition.cpp',
    'slang_entry_point.cpp',
    'slang_global_session.cpp',
    'slang_program.cpp',
)

if shaderc.found()
    merian_src += files('glsl_shader_compiler_shaderc.cpp')
else
    merian_src += files('glsl_shader_compiler_shaderc_stub.cpp')
endif
