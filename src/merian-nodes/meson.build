shader_include_dir = dir_source_root + '/src/merian-nodes'
merian_nodes_src = []
spv_compile_script = [dir_source_root + '/scripts/compile_shader.py']
glslc_args = ['-I' + dir_source_root + '/src', '-I' + shader_include_dir,'--target-env', 'vulkan1.3']

subdir('connectors')
subdir('graph')
subdir('nodes')

merian_nodes_lib = static_library(
    'merian-nodes',
    merian_nodes_src,
    dependencies: [
        fmt,
        glm,
        spdlog,
        vulkan,
        glfw,
    ],
    include_directories: inc_dirs,
    link_with: merian_lib,
    install : true
)
