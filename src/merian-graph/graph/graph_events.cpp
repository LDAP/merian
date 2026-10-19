#include "merian-graph/graph/graph.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace merian {

namespace {

bool matches_any(const std::vector<GraphEvent::Pattern>& patterns, const GraphEvent::Info& info) {
    return std::ranges::any_of(
        patterns, [&](const GraphEvent::Pattern& pattern) { return pattern.matches(info); });
}

} // namespace

void Graph::send_event(const std::string& event_name, const GraphEvent::Data& data) {
    if (event_name.empty() || event_name.find('/') != std::string::npos) {
        throw std::invalid_argument{fmt::format("invalid event name '{}'", event_name)};
    }
    send_event(GraphEvent::Info{"", "user", event_name}, data);
}

void Graph::register_event_listener_for_connect(const NodeHandle& node,
                                                const std::vector<GraphEvent::Pattern>& patterns,
                                                const GraphEvent::Listener& event_listener) {
    for (const GraphEvent::Pattern& pattern : patterns) {
        warn_if_no_sender(pattern);
    }
    event_listeners.push_back({node, patterns, event_listener});
}

void Graph::warn_if_no_sender(const GraphEvent::Pattern& pattern) const {
    const auto sends = [&](const std::string& node_name, const std::string& identifier) {
        return pattern.matches({node_name, identifier, pattern.event_name});
    };
    if (sends("", "user") || sends("", "graph") ||
        std::ranges::any_of(node_for_identifier, [&](const auto& entry) {
            return sends(node_data.at(entry.second).type_name, entry.first);
        })) {
        return;
    }
    SPDLOG_WARN("no node matches the event pattern '{}/{}/{}'", pattern.node_name,
                pattern.identifier, pattern.event_name);
}

void Graph::send_graph_event(const std::string& event_name, const GraphEvent::Data& data) {
    send_event(GraphEvent::Info{"", "graph", event_name}, data);
}

void Graph::send_event(const GraphEvent::Info& event_info, const GraphEvent::Data& data) {
    assert(!event_info.event_name.empty() && "event name cannot be empty.");
    assert(!event_info.identifier.empty() && "identifier cannot be empty.");
    assert((event_info.event_name.find('/') == event_info.event_name.npos) &&
           "event name cannot contain '/'.");

    SPDLOG_TRACE("sending event: {}/{}/{}", event_info.node_name, event_info.identifier,
                 event_info.event_name);

    std::string address =
        fmt::format("{}/{}/{}", event_info.node_name, event_info.identifier, event_info.event_name);
    if (!run_stop_event && matches_any(run_stop_patterns, event_info)) {
        run_stop_event = address;
    }
    recent_events[std::move(address)] = {run_iteration, std::chrono::steady_clock::now()};

    {
        // a removal erases listeners, so it waits until all ran
        const BusyScope busy_scope(*this);
        for (const EventListener& listener : event_listeners) {
            if (matches_any(listener.patterns, event_info)) {
                listener.listener(event_info, data);
            }
        }
    }
    apply_deferred_edits();
}

} // namespace merian
