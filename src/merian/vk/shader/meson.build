merian_src += shader_generator.process('fullscreen_triangle.vert')

merian_src += files(
    'shader_module.cpp',
    'shader_compiler_system_glslangValidator.cpp',
    'shader_compiler_system_glslc.cpp'
)

if shaderc.found()
    merian_src += files('shader_compiler_shaderc.cpp')
else
    merian_src += files('shader_compiler_shaderc_stub.cpp')
endif
