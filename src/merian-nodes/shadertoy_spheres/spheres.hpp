#include "merian-nodes/shadertoy/shadertoy.hpp"

namespace merian {

/* Ported version of https://www.shadertoy.com/view/lsX3DH
 *
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * @reindernijhoff
 *
 */
class ShadertoySpheres : public ShadertoyNode {

  public:
    ShadertoySpheres(const SharedContext context, const ResourceAllocatorHandle allocator);
};

} // namespace merian
