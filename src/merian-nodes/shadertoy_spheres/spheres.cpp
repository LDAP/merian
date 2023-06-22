#include "spheres.hpp"

namespace merian {

static const uint32_t spv[] = {
#include "spheres.comp.spv.h"
};

ShadertoySpheres::ShadertoySpheres(const SharedContext context,
                                   const ResourceAllocatorHandle allocator)
    : ShadertoyNode(context, allocator, sizeof(spv), spv) {}

} // namespace merian
