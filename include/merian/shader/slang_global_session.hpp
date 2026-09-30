#pragma once

#include "slang-com-ptr.h"
#include "slang.h"

#include "vulkan/vulkan.hpp"

#include <cstdint>
#include <string>

namespace merian {

vk::ShaderStageFlagBits vk_stage_for_slang_stage(const SlangStage slang_stage);

// Returns the global slang session.
Slang::ComPtr<slang::IGlobalSession> get_global_slang_session();

// Advanced whenever a source file changes; a session is reused only within one epoch.
uint64_t slang_source_epoch();
void bump_slang_source_epoch();

} // namespace merian
