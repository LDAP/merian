// Python.h must precede the standard headers
#include "bindings.hpp"

#include "merian-graph/graph/graph_description.hpp"
#include "merian-graph/graph/node_registry.hpp"
#include "merian-graph/merian_graph_extension.hpp"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace merian;

namespace {

std::vector<std::filesystem::path>
search_dirs_or_default(const std::optional<std::vector<std::filesystem::path>>& search_dirs) {
    return search_dirs.value_or(GraphDescription::default_search_dirs());
}

} // namespace

NB_MODULE(_graph, m) {
    const nb::module_ core = nb::module_::import_("merian._core");
    const nb::object graph_error =
        nb::exception<graph_errors::graph_error>(m, "GraphError", core.attr("MerianError"));
    nb::exception<graph_errors::config_error>(m, "ConfigError", graph_error);
    nb::exception<graph_errors::node_error>(m, "NodeError", graph_error);
    nb::exception<graph_errors::connector_error>(m, "ConnectorError", graph_error);

    m.attr("EXTENSIONS") = nb::make_tuple(MerianGraphExtension::name);

    python::main_thread_ident = nb::cast<unsigned long>(
        nb::module_::import_("threading").attr("main_thread")().attr("ident"));

    m.def(
        "node_types",
        [] {
            NodeRegistry& registry = NodeRegistry::get_instance();
            {
                const nb::gil_scoped_release release;
                registry.load_from_plugins();
            }
            nlohmann::json types = nlohmann::json::object();
            for (const auto& name : registry.node_names()) {
                types[name] = registry.node_info(name).description;
            }
            return types;
        },
        "{name: description} of what add_node takes: the node types and their variants, "
        "including plugins'.");
    m.def("default_search_dirs", &GraphDescription::default_search_dirs);
    m.def(
        "merge",
        [](nlohmann::json base, const nlohmann::json& overwrite) {
            GraphDescription::merge_into(base, overwrite);
            return base;
        },
        "base"_a, "overwrite"_a,
        "A deep merge of overwrite into base; \"$+$key\" appends to the array at key.");
    m.def(
        "find_config",
        [](const std::filesystem::path& path,
           const std::optional<std::vector<std::filesystem::path>>& search_dirs) {
            return GraphDescription::find_config(path, search_dirs_or_default(search_dirs));
        },
        "path"_a, "search_dirs"_a = nb::none());

    python::bind_description(m);
    python::bind_nodes(m);
    python::bind_window(m);
    python::bind_graph(m);
    python::bind_scene(m);
}
