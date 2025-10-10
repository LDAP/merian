merian_shaders_src = []

merian_shaders_lib = static_library(
    'merian-shaders',
    merian_shaders_src,
    dependencies: [
        fmt,
        glm,
        slang,
        spdlog,
        stb,
        vma,
        vulkan,
        vulkanheaders,
    ],
    include_directories: inc_dirs,
    link_with: merian_lib,
    install : true
)
