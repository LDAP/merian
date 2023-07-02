#include "node_io.hpp"

namespace merian {

NodeInputDescriptor::NodeInputDescriptor(const std::string& name,
                                         const vk::AccessFlags2 access_flags,
                                         const vk::PipelineStageFlags2 pipeline_stages,
                                         const uint32_t delay)
    : name(name), access_flags(access_flags), pipeline_stages(pipeline_stages), delay(delay) {}

NodeInputDescriptorImage::NodeInputDescriptorImage(const std::string& name,
                                                   const vk::AccessFlags2 access_flags,
                                                   const vk::PipelineStageFlags2 pipeline_stages,
                                                   const vk::ImageLayout required_layout,
                                                   const vk::ImageUsageFlags usage_flags,
                                                   const uint32_t delay,
                                                   const std::optional<SamplerHandle> sampler)
    : NodeInputDescriptor(name, access_flags, pipeline_stages, delay),
      required_layout(required_layout), usage_flags(usage_flags), sampler(sampler) {}

NodeInputDescriptorImage NodeInputDescriptorImage::compute_read(
    const std::string& name, const uint32_t delay, const std::optional<SamplerHandle> sampler) {
    return NodeInputDescriptorImage{name,
                                    vk::AccessFlagBits2::eShaderRead,
                                    vk::PipelineStageFlagBits2::eComputeShader,
                                    vk::ImageLayout::eShaderReadOnlyOptimal,
                                    vk::ImageUsageFlagBits::eSampled,
                                    delay,
                                    sampler};
}
NodeInputDescriptorImage NodeInputDescriptorImage::transfer_src(const std::string& name,
                                                                const uint32_t delay) {
    return NodeInputDescriptorImage{name,
                                    vk::AccessFlagBits2::eTransferRead,
                                    vk::PipelineStageFlagBits2::eTransfer,
                                    vk::ImageLayout::eTransferSrcOptimal,
                                    vk::ImageUsageFlagBits::eTransferSrc,
                                    delay};
}

NodeInputDescriptorBuffer::NodeInputDescriptorBuffer(const std::string& name,
                                                     const vk::AccessFlags2 access_flags,
                                                     const vk::PipelineStageFlags2 pipeline_stages,
                                                     const vk::BufferUsageFlags usage_flags,
                                                     const uint32_t delay)
    : NodeInputDescriptor(name, access_flags, pipeline_stages, delay), usage_flags(usage_flags) {}

NodeInputDescriptorBuffer NodeInputDescriptorBuffer::compute_read(const std::string& name) {
    return NodeInputDescriptorBuffer{
        name,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::BufferUsageFlagBits::eStorageBuffer,
    };
}

NodeInputDescriptorBuffer NodeInputDescriptorBuffer::transfer_src(const std::string& name) {
    return NodeInputDescriptorBuffer{
        name,
        vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::BufferUsageFlagBits::eTransferSrc,
    };
}

NodeOutputDescriptor::NodeOutputDescriptor(const std::string& name,
                                           const vk::AccessFlags2 access_flags,
                                           const vk::PipelineStageFlags2 pipeline_stages,
                                           const bool persistent)
    : name(name), access_flags(access_flags), pipeline_stages(pipeline_stages),
      persistent(persistent) {}

NodeOutputDescriptorImage::NodeOutputDescriptorImage(const std::string& name,
                                                     const vk::AccessFlags2 access_flags,
                                                     const vk::PipelineStageFlags2 pipeline_stages,
                                                     const vk::ImageCreateInfo create_info,
                                                     const vk::ImageLayout required_layout,
                                                     const bool persistent)
    : NodeOutputDescriptor(name, access_flags, pipeline_stages, persistent),
      create_info(create_info), required_layout(required_layout) {}

NodeOutputDescriptorImage NodeOutputDescriptorImage::compute_write(const std::string& name,
                                                                   const vk::Format format,
                                                                   const vk::Extent3D extent,
                                                                   const bool persistent) {
    const vk::ImageCreateInfo create_info{
        {},
        vk::ImageType::e2D,
        format,
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };

    return NodeOutputDescriptorImage{
        name,        vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader,
        create_info, vk::ImageLayout::eGeneral,         persistent};
}

NodeOutputDescriptorImage NodeOutputDescriptorImage::compute_write(const std::string& name,
                                                                   const vk::Format format,
                                                                   const uint32_t width,
                                                                   const uint32_t height,
                                                                   const bool persistent) {
    return compute_write(name, format, {width, height, 1}, persistent);
}

NodeOutputDescriptorImage NodeOutputDescriptorImage::transfer_write(const std::string& name,
                                                                    const vk::Format format,
                                                                    const vk::Extent3D extent,
                                                                    const bool persistent) {
    const vk::ImageCreateInfo create_info{
        {},
        vk::ImageType::e2D,
        format,
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };

    return NodeOutputDescriptorImage{
        name,        vk::AccessFlagBits2::eTransferWrite,  vk::PipelineStageFlagBits2::eTransfer,
        create_info, vk::ImageLayout::eTransferDstOptimal, persistent};
}

NodeOutputDescriptorImage NodeOutputDescriptorImage::transfer_write(const std::string& name,
                                                                    const vk::Format format,
                                                                    const uint32_t width,
                                                                    const uint32_t height,
                                                                    const bool persistent) {
    return transfer_write(name, format, {width, height, 1}, persistent);
}

NodeOutputDescriptorBuffer::NodeOutputDescriptorBuffer(
    const std::string& name,
    const vk::AccessFlags2 access_flags,
    const vk::PipelineStageFlags2 pipeline_stages,
    const vk::BufferCreateInfo create_info,
    const bool persistent)
    : NodeOutputDescriptor(name, access_flags, pipeline_stages, persistent),
      create_info(create_info) {}

} // namespace merian
