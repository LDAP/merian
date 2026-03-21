merian_src += shader_generator.process('imgui.vert',
                                       preserve_path_from: meson.current_source_dir())

merian_src += shader_generator.process('imgui.frag',
                                       preserve_path_from: meson.current_source_dir())

merian_src += files('imgui_context.cpp', 'imgui_renderer.cpp', 'imgui_input_listener.cpp', 'imgui_window_listener.cpp', 'imgui_merian_backend.cpp', 'imgui_merian_window_backend.cpp')
