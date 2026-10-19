gtest_main_dep = dependency('gtest', main : true, fallback: ['gtest', 'gtest_main_dep'])

test_shader_dir_arg = '-DTEST_SHADER_DIR="' + meson.current_source_dir() + '"'

test_timeouts = {
    'free_list': 30,
    'gbuffer_layout': 120,
    'geometry_utils': 30,
    'graph_connections': 120,
    'graph_data_flow': 300,
    'graph_edit': 300,
    'graph_in_place': 300,
    'graph_load_store': 120,
    'graph_memory': 120,
    'graph_run': 120,
    'graph_subgraphs': 300,
    'host_nodes': 120,
    'memory_allocator_tracking': 120,
    'memory_allocator_transient': 120,
    'node_properties': 120,
    'omm_bake': 60,
    'path_expression': 30,
    'path_debug_resample': 30,
    'properties_json_load': 30,
    'small_set': 30,
    'small_vector': 30,
    'svgf': 120,
    'versioned': 30,
}
foreach name, timeout : test_timeouts
    test(name, executable(
        'test-' + name.replace('_', '-'),
        'test_' + name + '.cpp',
        dependencies: [merian_dep, gtest_main_dep],
    ), timeout: timeout)
endforeach

foreach name : ['bsdf', 'material_system', 'scene', 'slang_hot_reload', 'texture_manager']
    test(name, executable(
        'test-' + name.replace('_', '-'),
        'test_' + name + '.cpp',
        dependencies: [merian_dep, gtest_main_dep],
        cpp_args: [test_shader_dir_arg],
    ), timeout: 120)
endforeach

test_slang_binding = executable(
    'test-slang-binding',
    'test_slang_binding.cpp',
    dependencies: [merian_dep, gtest_main_dep],
    cpp_args: ['-DSLANG_BINDING_TEST_SHADER_DIR="' + meson.current_source_dir() + '"'],
)
test('slang_binding', test_slang_binding, timeout: 120)

if get_option('glfw').enable_auto_if(glfw.found()).enabled() and host_machine.system() == 'linux'
    test_extension_glfw = executable(
        'test-extension-glfw',
        'test_extension_glfw.cpp',
        dependencies: [merian_dep, gtest_main_dep],
    )
    test('extension_glfw', test_extension_glfw, timeout: 60)
endif

if tgltf.found()
    test_model_dir_arg = '-DTEST_MODEL_DIR="' + meson.current_source_dir() + '/../subprojects/tinygltf/models"'

    test_gltf_scene = executable(
        'test-gltf-scene',
        'test_gltf_scene.cpp',
        dependencies: [merian_dep, gtest_main_dep],
        cpp_args: [test_shader_dir_arg, test_model_dir_arg],
    )
    test('gltf_scene', test_gltf_scene, timeout: 120)
endif

if pbrt_enabled
    test_pbrt_scene = executable(
        'test-pbrt-scene',
        'test_pbrt_scene.cpp',
        dependencies: [merian_dep, gtest_main_dep],
        cpp_args: [test_shader_dir_arg],
    )
    test('pbrt_scene', test_pbrt_scene, timeout: 120)
endif

if python_enabled
    foreach name, timeout : {'config': 60, 'graph': 120}
        test(
            'python_' + name,
            py,
            args: [files('python/test_' + name + '.py')],
            env: {'PYTHONPATH': python_package_dir},
            depends: [python_core_module, python_graph_module],
            timeout: timeout,
        )
    endforeach
endif
