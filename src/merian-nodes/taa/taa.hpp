#pragma once

#include "compute_node/compute_node.hpp"

namespace merian {

class TAANode : public ComputeNode {

  private:
    struct PushConstant {
        // higher value means more temporal reuse
        float temporal_alpha{.3};
    };

  public:
    TAANode(const SharedContext context, const ResourceAllocatorHandle allocator);

    virtual SpecializationInfoHandle get_specialization_info() const noexcept;

    virtual const void* get_push_constant();

    virtual std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept;

    virtual ShaderModuleHandle get_shader_module();

  private:
    PushConstant pc;
};

} // namespace merian
