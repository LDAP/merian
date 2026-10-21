#pragma once

// Python.h must precede the standard headers
#include "python_graph.hpp"

#include "merian-graph/nodes/host_source/host_buffer_source.hpp"
#include "merian-graph/nodes/host_source/host_image_source.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace merian::python {

using Seconds = std::variant<double, std::chrono::duration<double>>;

// Rounded. Raises OverflowError beyond the range of nanoseconds and for NaN.
std::optional<std::chrono::nanoseconds> to_nanoseconds(const std::optional<Seconds>& seconds);

// As to_nanoseconds for a timeout or duration: no limit beyond the range (inf), negative is zero.
std::optional<std::chrono::nanoseconds> to_limit(const std::optional<Seconds>& seconds);

struct GraphReference {
    nb::object graph;

    PythonGraph& python_graph() const;
};

template <typename NodeType> struct NodeReference : GraphReference {
    std::string identifier;
    mutable std::weak_ptr<NodeType> node;

    std::shared_ptr<NodeType> get() const {
        // a removed sink is cancelled, but may still be alive
        if (const auto current = node.lock()) {
            if constexpr (std::is_base_of_v<HostSink, NodeType>) {
                if (!current->is_cancelled()) {
                    return current;
                }
            } else {
                return current;
            }
        }
        std::shared_ptr<NodeType> current;
        if constexpr (std::is_base_of_v<HostSink, NodeType>) {
            current = python_graph().sink(identifier);
        } else if constexpr (std::is_base_of_v<HostSource, NodeType>) {
            current = std::dynamic_pointer_cast<NodeType>(python_graph().source(identifier));
            if (!current) {
                throw nb::key_error(
                    fmt::format("'{}' is a host source of another kind", identifier).c_str());
            }
        } else {
            current = python_graph().find<NodeType>(identifier, "node of this type");
        }
        node = current;
        return current;
    }
};

struct NodeByIdentifier : GraphReference {
    std::string identifier;

    // Raises KeyError once the graph no longer contains the node.
    const GraphHandle& graph_handle() const;
};

template <typename T> int traverse_graph_reference(PyObject* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));
    if (nb::inst_ready(self)) {
        Py_VISIT(nb::inst_ptr<T>(self)->graph.ptr());
    }
    return 0;
}

template <typename T> int clear_graph_reference(PyObject* self) {
    nb::inst_ptr<T>(self)->graph.reset();
    return 0;
}

template <typename T>
PyType_Slot graph_reference_slots[] = {
    {Py_tp_traverse, reinterpret_cast<void*>(traverse_graph_reference<T>)},
    {Py_tp_clear, reinterpret_cast<void*>(clear_graph_reference<T>)},
    {0, nullptr}};

template <typename NodeType>
nb::class_<NodeReference<NodeType>>
bind_node_reference(nb::module_& m, const char* name, const char* doc) {
    return nb::class_<NodeReference<NodeType>>(
               m, name, doc, nb::type_slots(graph_reference_slots<NodeReference<NodeType>>))
        .def_static("_find",
                    [name](const nb::object& graph, const std::string& identifier) {
                        NodeReference<NodeType> reference{{graph}, identifier, {}};
                        reference.node =
                            reference.python_graph().template find<NodeType>(identifier, name);
                        return reference;
                    })
        .def_ro("identifier", &NodeReference<NodeType>::identifier);
}

void bind_description(nb::module_& m);

void bind_nodes(nb::module_& m);

void bind_graph(nb::module_& m);

void bind_scene(nb::module_& m);

void bind_window(nb::module_& m);

} // namespace merian::python
