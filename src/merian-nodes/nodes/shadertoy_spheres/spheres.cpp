#include "spheres.hpp"

namespace merian_nodes {

static const uint32_t spv[] = {
#include "spheres.comp.spv.h"
};

ShadertoySpheresNode::ShadertoySpheresNode(const SharedContext context)
    : ShadertoyNode(context, sizeof(spv), spv) {}

} // namespace merian_nodes
