gtest_main_dep = dependency('gtest', main : true, fallback: ['gtest', 'gtest_main_dep'])

test_shader_dir_arg = '-DTEST_SHADER_DIR="' + meson.current_source_dir() + '"'

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
