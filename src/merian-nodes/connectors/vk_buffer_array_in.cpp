#include "vk_buffer_array_in.hpp"

#include "merian-nodes/graph/errors.hpp"

namespace merian_nodes {

BufferArrayIn::BufferArrayIn(const std::string& name,
                             const vk::ShaderStageFlags stage_flags,
                             const vk::AccessFlags2 access_flags,
                             const vk::PipelineStageFlags2 pipeline_stages)
    : TypedInputConnector(name, 0), stage_flags(stage_flags), access_flags(access_flags),
      pipeline_stages(pipeline_stages) {

    assert(access_flags && pipeline_stages);
    assert(!stage_flags || (access_flags & vk::AccessFlagBits2::eShaderRead));
}

std::optional<vk::DescriptorSetLayoutBinding> BufferArrayIn::get_descriptor_info() const {
    if (stage_flags)
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler,
                                              array_size, stage_flags, nullptr};
    return std::nullopt;
}

void BufferArrayIn::get_descriptor_update(const uint32_t binding,
                                          GraphResourceHandle& resource,
                                          DescriptorSetUpdate& update) {
    const auto& res = debugable_ptr_cast<BufferArrayResource>(resource);
    for (auto& pending_update : res->pending_updates) {
        const BufferHandle tex =
            res->buffers[pending_update] ? res->buffers[pending_update] : res->dummy_buffer;
        update.write_descriptor_buffer(binding, tex, 0, VK_WHOLE_SIZE, pending_update);
    }
}

const BufferArrayResource& BufferArrayIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<const BufferArrayResource>(resource);
}

void BufferArrayIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<BufferArrayOut>(output);
    if (!casted_output) {
        throw graph_errors::connector_error{
            fmt::format("BufferArrayIn {} cannot recive from {}.", name, output->name)};
    }
    array_size = casted_output->buffers.size();
}

BufferArrayInHandle BufferArrayIn::compute_read(const std::string& name) {
    return std::make_shared<BufferArrayIn>(name, vk::ShaderStageFlagBits::eCompute,
                                           vk::AccessFlagBits2::eShaderRead,
                                           vk::PipelineStageFlagBits2::eComputeShader);
}

} // namespace merian_nodes
