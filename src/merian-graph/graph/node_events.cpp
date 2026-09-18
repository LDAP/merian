#include "merian-graph/graph/node_events.hpp"

namespace merian {

void NodeEvents::register_listeners(const NodeIOLayout& io_layout) const {
    for (const Action& action : actions) {
        if (action.pattern.empty()) {
            continue;
        }
        io_layout.register_event_listener(
            action.pattern,
            [handler = action.handler](const GraphEvent::Info&, const GraphEvent::Data&) {
                handler();
                return false;
            });
    }
}

bool NodeEvents::properties(Properties& config) {
    bool changed = false;
    if (config.st_begin_child("events", "Events")) {
        for (Action& action : actions) {
            changed |= config.config_text(
                action.name, action.pattern, true,
                action.description.empty()
                    ? "Comma separated event patterns that trigger this action. Press enter to "
                      "confirm."
                    : action.description);
        }
        config.st_end_child();
    }
    return changed;
}

} // namespace merian
