python_graph_module = py.extension_module(
    '_graph',
    [
        'bind_description.cpp',
        'bind_graph.cpp',
        'bind_nodes.cpp',
        'bind_scene.cpp',
        'bind_window.cpp',
        'module.cpp',
        'python_graph.cpp',
        'python_properties.cpp',
        'transfer.cpp',
    ],
    dependencies: [merian_dep, nanobind],
    include_directories: include_directories('../..'),
    gnu_symbol_visibility: 'hidden',
    install: true,
    install_rpath: python_install_rpath,
    subdir: 'merian' / 'graph',
)

python_sources = ['__init__.py', 'config.py']
foreach source : python_sources
    fs.copyfile(source)
endforeach
py.install_sources(python_sources, subdir: 'merian' / 'graph')
