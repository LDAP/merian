#pragma once

#include "merian-nodes/graph/resource.hpp"
#include <fmt/format.h>

#include <any>

namespace merian_nodes {

class AnyResource : public GraphResource {
    friend class AnyOut;
    friend class AnyIn;

  public:
    AnyResource(const int32_t num_inputs) : num_inputs(num_inputs) {}

    void properties(merian::Properties& props) override {
        props.output_text(fmt::format("Type: {}", any.has_value() ? any.type().name() : "<empty>"));
    }

  private:
    const int32_t num_inputs;

    // reset after output, increased after input
    // if processed_inputs == num_inputs reset ptr if output is not persistent.
    int32_t processed_inputs;

    std::any any;
};

} // namespace merian_nodes
