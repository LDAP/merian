#pragma once

#include "merian-nodes/graph/resource.hpp"
#include <any>

namespace merian_nodes {

class AnyResource : public GraphResource {
    friend class AnyOut;
    friend class AnyIn;

  public:
    AnyResource(const int32_t num_inputs) : num_inputs(num_inputs) {}

  private:
    const int32_t num_inputs;

    // reset after output, increased after input
    // if processed_inputs == num_inputs reset ptr if output is not persistent.
    int32_t processed_inputs;

    std::any any;
};

} // namespace merian_nodes
