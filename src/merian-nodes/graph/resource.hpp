#pragma once

#include <memory>

namespace merian_nodes {

class GraphResource {
  public:
    virtual ~GraphResource(){};
};

using GraphResourceHandle = std::shared_ptr<GraphResource>;

} // namespace merian_nodes
