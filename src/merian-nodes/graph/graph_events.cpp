#include "merian-nodes/graph/graph.hpp"

#include <spdlog/spdlog.h>

namespace merian {

void Graph::send_event(const std::string& event_name,
                       const GraphEvent::Data& data,
                       const bool notify_all) {
    send_event(GraphEvent::Info{nullptr, "", "user", event_name}, data, notify_all);
}

void Graph::register_event_listener(const std::string& event_pattern,
                                    const GraphEvent::Listener& event_listener) {
    user_event_pattern_listener.push_back(std::make_pair(event_pattern, event_listener));
}

void Graph::register_event_listener_for_connect(const std::string& event_pattern,
                                                const GraphEvent::Listener& event_listener) {
    split(event_pattern, ",", [&](const std::string& split_pattern) {
        std::smatch match;
        if (!std::regex_match(split_pattern, match, EVENT_REGEX)) {
            SPDLOG_WARN("invalid event pattern '{}'", split_pattern);
            return;
        }
        const std::string& node_name = match[1];
        const std::string& node_identifier = match[2];
        const std::string& event_name = match[3];

        bool registered = false;
        if (node_name.empty()) {
            registered = true;
            if (node_identifier.empty()) {
                event_listeners["user"][event_name].emplace_back(event_listener);
                event_listeners["graph"][event_name].emplace_back(event_listener);
            } else if (node_identifier == "user" || node_identifier == "graph") {
                event_listeners[node_identifier][event_name].emplace_back(event_listener);
            } else {
                registered = false;
            }
        }
        for (const auto& [identifier, node] : node_for_identifier) {
            if ((node_name.empty() || registry.node_type_name(node) == node_name) &&
                (node_identifier.empty() || identifier == node_identifier)) {
                event_listeners[identifier][event_name].emplace_back(event_listener);
                registered = true;
            }
        }

        if (registered) {
            SPDLOG_DEBUG("registered listener for event pattern '{}'", split_pattern);
        } else {
            SPDLOG_WARN("no listener registered for event pattern '{}'. (no node type and node "
                        "identifier matched)",
                        split_pattern);
        }
    });
}

void Graph::send_graph_event(const std::string& event_name,
                             const GraphEvent::Data& data,
                             const bool notify_all) {
    send_event(GraphEvent::Info{nullptr, "", "graph", event_name}, data, notify_all);
}

void Graph::send_event(const GraphEvent::Info& event_info,
                       const GraphEvent::Data& data,
                       const bool notify_all) const {
    assert(!event_info.event_name.empty() && "event name cannot be empty.");
    assert(!event_info.identifier.empty() && "identifier cannot be empty.");
    assert((event_info.event_name.find('/') == event_info.event_name.npos) &&
           "event name cannot contain '/'.");

    SPDLOG_TRACE("sending event: {}/{}/{}, notify all={}", event_info.node_name,
                 event_info.identifier, event_info.event_name, notify_all);

    const auto identifier_it = event_listeners.find(event_info.identifier);
    if (identifier_it == event_listeners.end()) {
        return;
    }

    // exact match
    const auto event_it = identifier_it->second.find(event_info.event_name);
    if (event_it != identifier_it->second.end()) {
        if (notify_all) {
            for (const auto& listener : event_it->second) {
                listener(event_info, data);
            }
        } else {
            for (const auto& listener : event_it->second) {
                if (listener(event_info, data)) {
                    break;
                }
            }
        }
    }

    // any
    const auto event_any_it = identifier_it->second.find("");
    if (event_any_it != identifier_it->second.end()) {
        if (notify_all) {
            for (const auto& listener : event_any_it->second) {
                listener(event_info, data);
            }
        } else {
            for (const auto& listener : event_any_it->second) {
                if (listener(event_info, data)) {
                    break;
                }
            }
        }
    }
}

void Graph::set_on_run_starting(const std::function<void(GraphRun& graph_run)>& on_run_starting) {
    this->on_run_starting = on_run_starting;
}

void Graph::set_on_pre_submit(const std::function<void(GraphRun& graph_run)>& on_pre_submit) {
    this->on_pre_submit = on_pre_submit;
}

void Graph::set_on_post_submit(const std::function<void()>& on_post_submit) {
    this->on_post_submit = on_post_submit;
}

} // namespace merian
