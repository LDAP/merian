#pragma once

#include "merian-graph/nodes/guiding/guiding_node.hpp"
#include "merian-graph/nodes/render_pt_mcpg/mcpg_guiding_model.hpp"

namespace merian {

// Markov-chain path guiding over an adaptive hash grid, learning from an irradiance cache.
class MCPGGuidingNode : public GuidingNode {
  public:
    MCPGGuidingNode() : GuidingNode(std::make_shared<MCPGGuidingModel>()) {}
};

} // namespace merian
