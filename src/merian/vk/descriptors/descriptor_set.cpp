#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/command/command_buffer.hpp"

namespace merian {

void DescriptorSet::bind(const CommandBufferHandle& cmd,
                         const PipelineHandle& pipeline,
                         const uint32_t descriptor_set_index) const {
    cmd->bind_descriptor_set(pipeline, descriptor_set_index,
                             std::static_pointer_cast<const DescriptorSet>(shared_from_this()));
}

} // namespace merian
