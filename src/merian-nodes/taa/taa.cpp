#pragma once

#include "taa.hpp"

namespace merian {

TAANode::TAANode(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ComputeNode(context, allocator, sizeof(PushConstant)) {}

SpecializationInfoHandle TAANode::get_specialization_info() const noexcept {}

const void* TAANode::get_push_constant() {}

std::tuple<uint32_t, uint32_t, uint32_t> TAANode::get_group_count() const noexcept {}

ShaderModuleHandle TAANode::get_shader_module() {}

} // namespace merian
