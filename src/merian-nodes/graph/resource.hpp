#pragma once

#include "merian/utils/properties.hpp"

#include <memory>

namespace merian_nodes {

class GraphResource {
  public:
    virtual ~GraphResource(){};

    virtual void properties([[maybe_unused]] merian::Properties& props) {}
};

using GraphResourceHandle = std::shared_ptr<GraphResource>;

} // namespace merian_nodes
