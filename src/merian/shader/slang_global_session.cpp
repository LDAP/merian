#include "merian/shader/slang_global_session.hpp"

namespace {
Slang::ComPtr<slang::IGlobalSession> global_session;
}

namespace merian {

vk::ShaderStageFlagBits vk_stage_for_slang_stage(SlangStage slang_stage) {
    switch (slang_stage) {

    case SLANG_STAGE_NONE:
        throw std::invalid_argument{"stage cannot be none"};
    case SLANG_STAGE_VERTEX:
        return vk::ShaderStageFlagBits::eVertex;
    case SLANG_STAGE_HULL: // tessellation control
        return vk::ShaderStageFlagBits::eTessellationControl;
    case SLANG_STAGE_DOMAIN: // tessellation evaluation
        return vk::ShaderStageFlagBits::eTessellationEvaluation;
    case SLANG_STAGE_GEOMETRY:
        return vk::ShaderStageFlagBits::eGeometry;
    case SLANG_STAGE_FRAGMENT:
        return vk::ShaderStageFlagBits::eFragment;
    case SLANG_STAGE_COMPUTE:
        return vk::ShaderStageFlagBits::eCompute;
    case SLANG_STAGE_RAY_GENERATION:
        return vk::ShaderStageFlagBits::eRaygenKHR;
    case SLANG_STAGE_INTERSECTION:
        return vk::ShaderStageFlagBits::eIntersectionKHR;
    case SLANG_STAGE_ANY_HIT:
        return vk::ShaderStageFlagBits::eAnyHitKHR;
    case SLANG_STAGE_CLOSEST_HIT:
        return vk::ShaderStageFlagBits::eClosestHitKHR;
    case SLANG_STAGE_MISS:
        return vk::ShaderStageFlagBits::eMissKHR;
    case SLANG_STAGE_CALLABLE:
        return vk::ShaderStageFlagBits::eCallableKHR;
    case SLANG_STAGE_MESH:
        return vk::ShaderStageFlagBits::eMeshEXT;
    case SLANG_STAGE_AMPLIFICATION: // aka task shader
        return vk::ShaderStageFlagBits::eTaskEXT;

    // Not directly mappable
    case SLANG_STAGE_DISPATCH:
    case SLANG_STAGE_COUNT:
    default:
        throw std::invalid_argument{"stage not supported."};
    }
}

Slang::ComPtr<slang::IGlobalSession> get_global_slang_session() {
    if (global_session.get() == nullptr) {
        createGlobalSession(global_session.writeRef());
    }
    return global_session;
}

} // namespace merian
