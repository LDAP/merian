// Python.h must precede the standard headers
#include "bindings.hpp"
#include "transfer.hpp"

#include <algorithm>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

namespace merian::python {

using namespace nb::literals;

PythonGraph& GraphReference::python_graph() const {
    return nb::cast<PythonGraph&>(graph);
}

const GraphHandle& NodeByIdentifier::graph_handle() const {
    const GraphHandle& handle = python_graph().editable_handle();
    if (!handle->find_node_for_identifier(identifier)) {
        throw nb::key_error(fmt::format("no node '{}'", identifier).c_str());
    }
    return handle;
}

namespace {

std::chrono::duration<double> to_duration(const Seconds& seconds) {
    return std::visit([](const auto value) { return std::chrono::duration<double>(value); },
                      seconds);
}

std::chrono::nanoseconds round_to_nanoseconds(const std::chrono::duration<double> seconds) {
    // NaN fails the comparison too
    if (!(std::chrono::abs(seconds) < std::chrono::nanoseconds::max())) {
        throw std::overflow_error(fmt::format("{} s is out of range", seconds.count()));
    }
    return std::chrono::round<std::chrono::nanoseconds>(seconds);
}

} // namespace

std::optional<std::chrono::nanoseconds> to_nanoseconds(const std::optional<Seconds>& seconds) {
    if (!seconds) {
        return std::nullopt;
    }
    return round_to_nanoseconds(to_duration(*seconds));
}

std::optional<std::chrono::nanoseconds> to_limit(const std::optional<Seconds>& seconds) {
    if (!seconds) {
        return std::nullopt;
    }
    const std::chrono::duration<double> limit = to_duration(*seconds);
    if (limit >= std::chrono::nanoseconds::max()) {
        return std::nullopt;
    }
    return round_to_nanoseconds(std::max(limit, std::chrono::duration<double>::zero()));
}

void bind_nodes(nb::module_& m) {
    nb::enum_<NodeState>(m, "NodeState")
        .value("OK", NodeState::OK)
        .value("DISABLED", NodeState::DISABLED)
        .value("BYPASSED", NodeState::BYPASSED)
        .value("UNSUPPORTED", NodeState::UNSUPPORTED)
        .value("FORCE_DISABLED", NodeState::FORCE_DISABLED)
        .value("ERRORED", NodeState::ERRORED);

    nb::enum_<NodeConnectors::Output::Kind>(m, "ConnectorKind")
        .value("IMAGE", NodeConnectors::Output::Kind::IMAGE)
        .value("BUFFER", NodeConnectors::Output::Kind::BUFFER)
        .value("OTHER", NodeConnectors::Output::Kind::OTHER);

    nb::class_<NodeConnectors::Input>(m, "InputConnector")
        .def_ro("name", &NodeConnectors::Input::name)
        .def_ro("delay", &NodeConnectors::Input::delay, "Iterations it reads back.")
        .def_ro("optional", &NodeConnectors::Input::optional)
        .def("__repr__", [](const NodeConnectors::Input& input) {
            return fmt::format("InputConnector('{}', delay={}, optional={})", input.name,
                               input.delay, input.optional);
        });

    nb::class_<NodeConnectors::Output>(m, "OutputConnector")
        .def_ro("name", &NodeConnectors::Output::name)
        .def_ro("kind", &NodeConnectors::Output::kind)
        .def("__repr__", [](const NodeConnectors::Output& output) {
            return fmt::format("OutputConnector('{}', {})", output.name,
                               nb::cast<std::string>(nb::str(nb::cast(output.kind))));
        });

    nb::class_<NodeByIdentifier>(m, "Node",
                                 "A node of a graph; raises KeyError once the graph removed it.",
                                 nb::type_slots(graph_reference_slots<NodeByIdentifier>))
        .def_ro("identifier", &NodeByIdentifier::identifier)
        .def_prop_ro("type",
                     [](const NodeByIdentifier& node) {
                         return node.graph_handle()->node_get_type(node.identifier);
                     })
        .def_prop_ro("state",
                     [](const NodeByIdentifier& node) {
                         return node.graph_handle()->node_get_state(node.identifier);
                     })
        .def_prop_ro(
            "inputs",
            [](const NodeByIdentifier& node) {
                return node.graph_handle()->node_get_connectors(node.identifier).inputs;
            },
            "As the last connect described them: declared, then the public ones of a subgraph.")
        .def_prop_ro(
            "outputs",
            [](const NodeByIdentifier& node) {
                return node.graph_handle()->node_get_connectors(node.identifier).outputs;
            },
            "As the last connect described them, once the inputs are satisfied: declared, then "
            "the public ones of a subgraph.")
        .def_prop_ro(
            "connections",
            [](const NodeByIdentifier& node) {
                return node.graph_handle()->node_get_connectors(node.identifier).connections;
            },
            "{input: (node, output)} as edited, also to connectors that are not described.")
        .def(
            "properties",
            [](const NodeByIdentifier& node, const nb::object& ui) {
                std::ignore = node.graph_handle();
                node.python_graph().node_properties(node.identifier, ui);
            },
            "ui"_a,
            "Records whether it is enabled, its linearization order, properties, connectors, "
            "resources and subnodes into ui, a merian.Properties; what ui returns as set applies "
            "now, see Graph.needs_reconnect.")
        .def("get_properties",
             [](const NodeByIdentifier& node) {
                 return node.graph_handle()->node_get_properties(node.identifier);
             })
        .def(
            "set_properties",
            [](const NodeByIdentifier& node, const nlohmann::json& properties) {
                const GraphHandle graph_handle = node.graph_handle();
                node.python_graph().without_gil("setting properties", [&] {
                    graph_handle->node_set_properties(node.identifier, properties);
                });
            },
            "properties"_a,
            "Sets the given properties, the others keep their value. Raises ValueError for "
            "unknown keys and invalid values; the valid ones stay set.")
        .def("__repr__", [](const NodeByIdentifier& node) {
            return fmt::format("Node('{}')", node.identifier);
        });

    nb::class_<HostSink::Capture>(m, "Capture",
                                  "An iteration's input of a host sink, in host memory.")
        .def_ro("run_iteration", &HostSink::Capture::run_iteration,
                "Iterations since the last connect, see Graph.run_iteration.")
        .def_ro("time", &HostSink::Capture::time, "Graph time in seconds.")
        .def_prop_ro(
            "format",
            [](const HostSink::Capture& capture) -> std::optional<std::string> {
                if (capture.copy.format == vk::Format::eUndefined) {
                    return std::nullopt;
                }
                return vk::to_string(capture.copy.format);
            },
            "None for a buffer.")
        .def_prop_ro(
            "extent",
            [](const HostSink::Capture& capture) -> nb::object {
                if (capture.copy.format == vk::Format::eUndefined) {
                    return nb::none();
                }
                return nb::make_tuple(capture.copy.extent.width, capture.copy.extent.height,
                                      capture.copy.extent.depth);
            },
            "(width, height, depth), None for a buffer.")
        .def_prop_ro("nbytes", [](const HostSink::Capture& capture) { return capture.copy.size; })
        .def_prop_ro(
            "array", [](const HostSink::Capture& capture) { return host_copy_array(capture.copy); },
            nb::rv_policy::reference,
            "The data without a copy: texels in the order of format, (height, width, channels) "
            "or (depth, height, width, channels), or nbytes bytes for a buffer.")
        .def(
            "__array__",
            [](const HostSink::Capture& capture, const nb::handle dtype,
               const std::optional<bool> copy) {
                nb::object array =
                    nb::cast(host_copy_array(capture.copy), nb::rv_policy::reference);
                if (!dtype.is_none()) {
                    array = array.attr("astype")(dtype);
                } else if (copy.value_or(false)) {
                    array = array.attr("copy")();
                }
                return array;
            },
            "dtype"_a = nb::none(), "copy"_a = nb::none())
        .def("__repr__", [](const HostSink::Capture& capture) {
            if (capture.copy.format == vk::Format::eUndefined) {
                return fmt::format("Capture(run_iteration={}, {} bytes)", capture.run_iteration,
                                   capture.copy.size);
            }
            return fmt::format("Capture(run_iteration={}, {}x{}x{}, {})", capture.run_iteration,
                               capture.copy.extent.width, capture.copy.extent.height,
                               capture.copy.extent.depth, vk::to_string(capture.copy.format));
        });

    using SinkReference = NodeReference<HostSink>;
    nb::class_<SinkReference>(
        m, "HostSink",
        "Captures of a Host Image or Buffer Sink in the iterations asked for, which wait for "
        "get() and get_all(), e.g. from an after_iteration callback of Graph.run. All methods "
        "are callable from any thread, except while the graph runs on another thread after it "
        "removed or reloaded the sink, which is looked up again by identifier.",
        nb::type_slots(graph_reference_slots<SinkReference>))
        .def(
            "capture",
            [](const SinkReference& sink, const uint32_t count) { sink.get()->capture(count); },
            "count"_a = 1)
        .def(
            "capture_at",
            [](const SinkReference& sink, const std::vector<uint64_t>& iterations) {
                sink.get()->capture_at(iterations);
            },
            "iterations"_a, "Captures these run iterations (see Graph.run_iteration).")
        .def(
            "capture_every",
            [](const SinkReference& sink, const uint32_t every) {
                sink.get()->capture_every(every);
            },
            "every"_a = 1,
            "Captures every n-th run iteration (see Graph.run_iteration), 0 stops. Skips an "
            "iteration while more captures wait than iterations are in flight.")
        .def(
            "get",
            [](const SinkReference& sink, const std::optional<Seconds> timeout) {
                const std::shared_ptr<HostSink> node = sink.get();
                std::optional<HostSink::Capture> capture;
                wait_interruptibly(
                    [&](const std::chrono::nanoseconds slice) {
                        capture = node->get(slice);
                        return capture.has_value() || node->is_cancelled();
                    },
                    to_limit(timeout));
                return capture;
            },
            "timeout"_a = nb::none(),
            "The oldest capture once its data arrived, waiting up to timeout (seconds or "
            "timedelta; forever if None, not at all if 0) for it; None on timeout or once the "
            "graph closed, reloaded or removed the sink. The graph does not wait for consumer "
            "threads.")
        .def(
            "get_all", [](const SinkReference& sink) { return sink.get()->get_all(); },
            "The captures whose data arrived, oldest first, without waiting.")
        .def("__repr__", [](const SinkReference& sink) {
            return fmt::format("HostSink('{}')", sink.identifier);
        });

    using ImageSourceReference = NodeReference<HostImageSource>;
    nb::class_<ImageSourceReference>(m, "HostImageSource",
                                     nb::type_slots(graph_reference_slots<ImageSourceReference>))
        .def(
            "upload",
            [](const ImageSourceReference& source, const HostArray& array, const bool srgb) {
                upload_image(*source.get(), array, srgb);
            },
            "array"_a, "srgb"_a = false,
            "Output from the next iteration that starts after the upload, (height, width[, "
            "channels]); a new size or format reconnects the graph first. Callable from any "
            "thread.");

    using BufferSourceReference = NodeReference<HostBufferSource>;
    nb::class_<BufferSourceReference>(m, "HostBufferSource",
                                      nb::type_slots(graph_reference_slots<BufferSourceReference>))
        .def(
            "upload",
            [](const BufferSourceReference& source, const HostArray& array) {
                upload_bytes(*source.get(), array);
            },
            "array"_a,
            "Outputs the bytes of array from the next iteration that starts after the upload; a "
            "new size reconnects the graph first. Callable from any thread.");
}

} // namespace merian::python
