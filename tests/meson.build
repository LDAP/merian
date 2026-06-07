gtest_main_dep = dependency('gtest', main : true, fallback: ['gtest', 'gtest_main_dep'])

test_shader_dir_arg = '-DTEST_SHADER_DIR="' + meson.current_source_dir() + '"'

test_free_list = executable(
    'test-free-list',
    'test_free_list.cpp',
    dependencies: [merian_dep, gtest_main_dep],
)
test('free_list', test_free_list, timeout: 30)

test_small_vector = executable(
    'test-small-vector',
    'test_small_vector.cpp',
    dependencies: [merian_dep, gtest_main_dep],
)
test('small_vector', test_small_vector, timeout: 30)

test_small_set = executable(
    'test-small-set',
    'test_small_set.cpp',
    dependencies: [merian_dep, gtest_main_dep],
)
test('small_set', test_small_set, timeout: 30)

test_versioned = executable(
    'test-versioned',
    'test_versioned.cpp',
    dependencies: [merian_dep, gtest_main_dep],
)
test('versioned', test_versioned, timeout: 30)

test_slang_binding = executable(
    'test-slang-binding',
    'test_slang_binding.cpp',
    dependencies: [merian_dep, gtest_main_dep],
    cpp_args: ['-DSLANG_BINDING_TEST_SHADER_DIR="' + meson.current_source_dir() + '"'],
)
test('slang_binding', test_slang_binding, timeout: 120)

test_texture_manager = executable(
    'test-texture-manager',
    'test_texture_manager.cpp',
    dependencies: [merian_dep, gtest_main_dep],
    cpp_args: [test_shader_dir_arg],
)
test('texture_manager', test_texture_manager, timeout: 120)

test_material_system = executable(
    'test-material-system',
    'test_material_system.cpp',
    dependencies: [merian_dep, gtest_main_dep],
    cpp_args: [test_shader_dir_arg],
)
test('material_system', test_material_system, timeout: 120)

test_scene = executable(
    'test-scene',
    'test_scene.cpp',
    dependencies: [merian_dep, gtest_main_dep],
    cpp_args: [test_shader_dir_arg],
)
test('scene', test_scene, timeout: 120)

test_slang_hot_reload = executable(
    'test-slang-hot-reload',
    'test_slang_hot_reload.cpp',
    dependencies: [merian_dep, gtest_main_dep],
    cpp_args: [test_shader_dir_arg],
)
test('slang_hot_reload', test_slang_hot_reload, timeout: 120)

test_bsdf = executable(
    'test-bsdf',
    'test_bsdf.cpp',
    dependencies: [merian_dep, gtest_main_dep],
    cpp_args: [test_shader_dir_arg],
)
test('bsdf', test_bsdf, timeout: 120)

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
