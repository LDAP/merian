gtest_dep = dependency('gtest', main: true, required: true)
gtest_main_dep = dependency('gtest_main', required: true)

test_exe = executable('merian_tests',
                      'test_example.cpp',
                      dependencies: [gtest_dep, gtest_main_dep, merian_dep]
)

test('Merian tests', test_exe)