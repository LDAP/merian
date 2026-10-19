// Python.h must precede the standard headers
#include "bindings.hpp"

#include <algorithm>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace merian::python {

using namespace nb::literals;

namespace {

std::pair<std::string, std::string> split_address(const GraphDescription& description,
                                                  const std::string& address) {
    const std::size_t dot = address.find('.');
    if (dot == std::string::npos) {
        throw std::invalid_argument(fmt::format("expected 'node.connector', got '{}'", address));
    }
    const std::string node = address.substr(0, dot);
    if (!description.has_node(node)) {
        throw std::invalid_argument(fmt::format("no node '{}' for '{}'", node, address));
    }
    return {node, address.substr(dot + 1)};
}

std::string add_sink(GraphDescription& description,
                     const std::string& node_type,
                     const std::string& output,
                     const std::optional<std::string>& identifier) {
    const auto [node, connector] = split_address(description, output);
    const std::string sink = description.add_node(node_type, identifier);
    description.add_connection(node, sink, connector, "src");
    return sink;
}

std::string add_source(GraphDescription& description,
                       const std::string& node_type,
                       const std::string& input,
                       const std::optional<std::string>& identifier) {
    const auto [node, connector] = split_address(description, input);
    const std::string source = description.add_node(node_type, identifier);
    description.add_connection(source, node, "out", connector);
    return source;
}

} // namespace

void bind_description(nb::module_& m) {
    nb::class_<GraphDescription>(m, "Description",
                                 "Nodes, connections and their configuration; no GPU objects.")
        .def(nb::init<>())
        .def_static("from_dict", &GraphDescription::from_json, "config"_a)
        .def("to_dict", &GraphDescription::to_json)
        .def_prop_ro(
            "nodes",
            [](const GraphDescription& description) {
                nlohmann::json types = nlohmann::json::object();
                for (const auto& [identifier, info] : description.get_nodes()) {
                    types[identifier] = info.node_type;
                }
                return types;
            },
            "{identifier: node type}")
        .def_prop_ro(
            "connections",
            [](const GraphDescription& description) {
                std::vector<std::tuple<std::string, std::string, std::string, std::string>>
                    connections;
                for (const auto& [dst, info] : description.get_nodes()) {
                    for (const auto& [dst_input, source] : info.incoming_connections) {
                        connections.emplace_back(source.first, dst, source.second, dst_input);
                    }
                }
                std::ranges::sort(connections);
                return connections;
            },
            "[(src, dst, src_output, dst_input)]")
        .def_prop_rw("graph_properties", &GraphDescription::get_graph_properties,
                     &GraphDescription::set_graph_properties)
        .def("add_node", &GraphDescription::add_node, "node_type"_a, "identifier"_a = nb::none(),
             "properties"_a = nlohmann::json::object(), "Returns the identifier.")
        .def("remove_node", &GraphDescription::remove_node, "identifier"_a)
        .def("add_connection", &GraphDescription::add_connection, "src"_a, "dst"_a, "src_output"_a,
             "dst_input"_a)
        .def("remove_connection", &GraphDescription::remove_connection, "src"_a, "dst"_a,
             "dst_input"_a)
        .def(
            "add_image_sink",
            [](GraphDescription& description, const std::string& output,
               const std::optional<std::string>& identifier) {
                return add_sink(description, "Host Image Sink", output, identifier);
            },
            "output"_a, "identifier"_a = nb::none(),
            "Adds a Host Image Sink fed by 'node.output'; it captures nothing until asked (see "
            "HostSink). Returns its identifier.")
        .def(
            "add_buffer_sink",
            [](GraphDescription& description, const std::string& output,
               const std::optional<std::string>& identifier) {
                return add_sink(description, "Host Buffer Sink", output, identifier);
            },
            "output"_a, "identifier"_a = nb::none(),
            "Adds a Host Buffer Sink fed by 'node.output'. Returns its identifier.")
        .def(
            "add_image_source",
            [](GraphDescription& description, const std::string& input,
               const std::optional<std::string>& identifier) {
                return add_source(description, "Host Image Source", input, identifier);
            },
            "input"_a, "identifier"_a = nb::none(),
            "Adds a Host Image Source that feeds 'node.input'. Returns its identifier.")
        .def(
            "add_buffer_source",
            [](GraphDescription& description, const std::string& input,
               const std::optional<std::string>& identifier) {
                return add_source(description, "Host Buffer Source", input, identifier);
            },
            "input"_a, "identifier"_a = nb::none(),
            "Adds a Host Buffer Source that feeds 'node.input'. Returns its identifier.");
}

} // namespace merian::python
