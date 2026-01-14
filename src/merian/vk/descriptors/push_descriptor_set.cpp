#include "merian/vk/descriptors/push_descriptor_set.hpp"
#include "merian/vk/command/command_buffer.hpp"

namespace merian {

void PushDescriptorSet::bind(const CommandBufferHandle& cmd,
                             const PipelineHandle& pipeline,
                             const uint32_t descriptor_set_index) const {
    cmd->push_descriptor_set(pipeline, descriptor_set_index,
                             std::static_pointer_cast<const PushDescriptorSet>(shared_from_this()));
}

} // namespace merian
