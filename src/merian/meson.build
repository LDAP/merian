merian_src = files(
    'io/file_loader.cpp',
    'io/tinyobj.cpp',
    'utils/audio/audio_device.cpp',
    'utils/audio/sdl_audio_device.cpp',
    'utils/camera/camera.cpp',
    'utils/camera/camera_animator.cpp',
    'utils/concurrent/thread_pool.cpp',
    'utils/input_controller_dummy.cpp',
    'utils/input_controller_glfw.cpp',
    'utils/normal_encoding.cpp',
    'utils/properties_imgui.cpp',
    'utils/properties_json_dump.cpp',
    'utils/properties_json_load.cpp',
    'utils/renderdoc.cpp',
    'utils/stopwatch.cpp',
    'vk/command/command_pool.cpp',
    'vk/command/queue.cpp',
    'vk/command/ring_command_pool.cpp',
    'vk/context.cpp',
    'vk/extension/extension.cpp',
    'vk/extension/extension_resources.cpp',
    'vk/extension/extension_vk_debug_utils.cpp',
    'vk/extension/extension_vk_glfw.cpp',
    'vk/memory/buffer_suballocator.cpp',
    'vk/memory/memory_allocator.cpp',
    'vk/memory/memory_allocator_vma.cpp',
    'vk/memory/resource_allocations.cpp',
    'vk/memory/resource_allocator.cpp',
    'vk/memory/staging_memory_manager.cpp',
    'vk/pipeline/pipeline_graphics_builder.cpp',
    'vk/raytrace/as_compressor.cpp',
    'vk/raytrace/as_builder_blas.cpp',
    'vk/raytrace/as_builder_tlas.cpp',
    'vk/renderpass/renderpass.cpp',
    'vk/renderpass/renderpass_builder.cpp',
    'vk/sampler/sampler_pool.cpp',
    'vk/shader/shader_compiler.cpp',
    'vk/shader/shader_hotreloader.cpp',
    'vk/sync/semaphore_timeline.cpp',
    'vk/utils/barriers.cpp',
    'vk/utils/vk_enums.cpp',
    'vk/utils/profiler.cpp',
    'vk/window/glfw_imgui.cpp',
    'vk/window/glfw_window.cpp',
    'vk/window/swapchain.cpp',
)

if shaderc.found()
    merian_src += files('vk/shader/shader_compiler_shaderc.cpp')
endif

subdir('vk/shader')

merian_lib = static_library(
    'merian',
    [merian_src],
    dependencies: [
        fmt,
        glfw,
        glm,
        imgui,
        nlohmann_json,
        sdl2,
        shaderc,
        spdlog,
        tgltf,
        tol,
        vma,
        vulkan,
    ],
    include_directories: inc_dirs,
    install : true
)
