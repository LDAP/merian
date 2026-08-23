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

// Monotonic counter advanced whenever a module source changes. A session can only be reused while
// the epoch is unchanged, because a slang::ISession binds each module name to one immutable source.
uint64_t slang_source_epoch();
void bump_slang_source_epoch();

// Records the source a module name is compiled from and advances the epoch when the name is bound
// to a different one, so the binding is never observed by a session that already holds the old one.
void bind_slang_module_source(const std::string& name, uint64_t source_hash);

} // namespace merian
