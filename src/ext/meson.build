ext_src = files(
    'stb_image.cpp',
    'stb_image_write.cpp',
)

ext_lib = static_library(
    'merian-ext',
    [ext_src],
    dependencies: [
        vulkan,
        glfw
    ],
    include_directories: [inc_dirs],
    install : true
)
