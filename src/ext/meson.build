ext_src = files(
    'stb_image.cpp',
    'stb_image_write.cpp',
    'vma.cpp',

    'imgui/imgui.cpp',
    'imgui/imgui_demo.cpp',
    'imgui/imgui_draw.cpp',
    'imgui/imgui_tables.cpp',
    'imgui/imgui_widgets.cpp',
    'imgui/backends/imgui_impl_vulkan.cpp',
    'imgui/backends/imgui_impl_glfw.cpp',
)

ext_lib = static_library(
    'merian-ext',
    [ext_src],
    dependencies: [
        vulkan
    ],
    include_directories: [inc_dirs],
    install : true
)
