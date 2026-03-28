gtest_main_dep = dependency('gtest', main : true, fallback: ['gtest', 'gtest_main_dep'])

test_slang_binding = executable(
    'test-slang-binding',
    'test_slang_binding.cpp',
    dependencies: [merian_dep, gtest_main_dep],
    cpp_args: ['-DSLANG_BINDING_TEST_SHADER_DIR="' + meson.current_source_dir() + '"'],
)
test('slang_binding', test_slang_binding, timeout: 120)
