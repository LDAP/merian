#pragma once

#include "slang-com-ptr.h"
#include "slang.h"

#include "vulkan/vulkan.hpp"

#include <cstdint>

namespace merian {

vk::ShaderStageFlagBits vk_stage_for_slang_stage(const SlangStage slang_stage);

// Returns the global slang session.
Slang::ComPtr<slang::IGlobalSession> get_global_slang_session();

// Monotonic counter advanced whenever a module source changes. A session can only be reused while
// the epoch is unchanged, because a slang::ISession binds each module name to one immutable source.
uint64_t slang_source_epoch();
void bump_slang_source_epoch();

} // namespace merian
