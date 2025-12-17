#pragma once

#include "merian-nodes/graph/node.hpp"

namespace merian {

class GBufferRTNode : public Node {
    NodeStatusFlags
    on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                 [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) override {
        return {};
    }
};

} // namespace merian
