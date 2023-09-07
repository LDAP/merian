shader_include_dir = dir_source_root + '/src/merian-nodes'
merian_nodes_src = []
spv_compile_script = [dir_source_root + '/scripts/compile_shader.py']
glslc_args = ['-I' + dir_source_root + '/src', '-I' + shader_include_dir,'--target-env', 'vulkan1.3']

subdir('add')
subdir('ab_compare')
subdir('accumulate')
subdir('bloom')
subdir('color_output')
subdir('compute_node')
subdir('exposure')
subdir('image')
subdir('image_write')
subdir('mean')
subdir('median_approx')
subdir('shadertoy')
subdir('shadertoy_spheres')
subdir('svgf')
subdir('taa')
subdir('tonemap')
subdir('vkdt_filmcurv')

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
