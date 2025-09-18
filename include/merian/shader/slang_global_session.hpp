#pragma once

#include "slang-com-ptr.h"
#include "slang.h"

#include "vulkan/vulkan.hpp"

namespace merian {

vk::ShaderStageFlagBits vk_stage_for_slang_stage(const SlangStage slang_stage);

// Returns the global slang session.
Slang::ComPtr<slang::IGlobalSession> get_global_slang_session();

}
