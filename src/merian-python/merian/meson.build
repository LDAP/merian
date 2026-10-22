python_core_module = py.extension_module(
    '_core',
    ['core.cpp', 'log_sink.cpp'],
    dependencies: [merian_dep, nanobind],
    include_directories: include_directories('..'),
    gnu_symbol_visibility: 'hidden',
    install: true,
    install_rpath: python_install_rpath,
    subdir: 'merian',
)

python_sources = ['__init__.py', 'properties.py']
foreach source : python_sources
    fs.copyfile(source)
endforeach
py.install_sources(python_sources, subdir: 'merian')

# the DLL directories of the build tree, which __init__.py adds without meson devenv
if host_machine.system() == 'windows'
    custom_target(
        'devenv',
        output: 'devenv.env',
        command: [find_program('meson'), 'devenv', '-C', meson.project_build_root(), '--dump',
                  '@OUTPUT@', '--dump-format', 'vscode'],
        depend_files: meson.project_build_root() / 'meson-info' / 'meson-info.json',
        build_by_default: true,
    )
endif

subdir('graph')
