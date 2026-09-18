#pragma once

#include "merian-graph/graph/node_io.hpp"
#include "merian/utils/properties.hpp"

#include <functional>
#include <string>
#include <vector>

namespace merian {

// The event patterns a node listens on, one per action it offers.
class NodeEvents {
  public:
    struct Action {
        std::string name;
        std::function<void()> handler;
        std::string pattern{};
        std::string description{};
    };

    explicit NodeEvents(std::vector<Action> actions) : actions(std::move(actions)) {}

    void register_listeners(const NodeIOLayout& io_layout) const;

    // Returns true if a pattern changed, which needs a reconnect to take effect.
    [[nodiscard]] bool properties(Properties& config);

  private:
    std::vector<Action> actions;
};

} // namespace merian
