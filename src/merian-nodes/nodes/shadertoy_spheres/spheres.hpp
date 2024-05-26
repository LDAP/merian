#pragma once

#include "merian-nodes/nodes/shadertoy/shadertoy.hpp"

namespace merian_nodes {

/* Ported version of https://www.shadertoy.com/view/lsX3DH
 *
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * @reindernijhoff
 *
 */
class ShadertoySpheresNode : public ShadertoyNode {

  public:
    ShadertoySpheresNode(const SharedContext context);
};

} // namespace merian_nodes