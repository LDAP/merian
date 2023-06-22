#pragma once

#include "merian/vk/memory/resource_allocations.hpp"

#include "vulkan/vulkan.hpp"

#include <functional>
#include <memory>

namespace merian {
struct NodeInputDescriptor {
  public:
    NodeInputDescriptor(const std::string& name,
                        const vk::AccessFlags2 access_flags,
                        const vk::PipelineStageFlags2 pipeline_stages,
                        const uint32_t delay)
        : name(name), access_flags(access_flags), pipeline_stages(pipeline_stages), delay(delay) {}
    std::string name;
    // The types of access on this input, only reads are allowed
    vk::AccessFlags2 access_flags;
    // The pipeline stages that access this input
    vk::PipelineStageFlags2 pipeline_stages;
    // The number of iterations to the delay the output to this input.
    // For example, 0 means the most current output, 1 means delayed by one frame, and so on.
    // Note setting this to n leads to allocation of at least n copies of the resource.
    uint32_t delay;
};

struct NodeInputDescriptorImage : public NodeInputDescriptor {
  public:
    NodeInputDescriptorImage(const std::string& name,
                             const vk::AccessFlags2 access_flags,
                             const vk::PipelineStageFlags2 pipeline_stages,
                             const vk::ImageLayout required_layout,
                             const vk::ImageUsageFlags usage_flags,
                             const uint32_t delay = 0)
        : NodeInputDescriptor(name, access_flags, pipeline_stages, delay),
          required_layout(required_layout), usage_flags(usage_flags) {}

    vk::ImageLayout required_layout;
    vk::ImageUsageFlags usage_flags;

    static NodeInputDescriptorImage compute_read(const std::string& name) {
        return NodeInputDescriptorImage{
            name,
            vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageUsageFlagBits::eSampled,
        };
    }
    static NodeInputDescriptorImage transfer_src(const std::string& name, const uint32_t delay = 0) {
        return NodeInputDescriptorImage{
            name,
            vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageUsageFlagBits::eTransferSrc,
            delay
        };
    }
};

struct NodeInputDescriptorBuffer : public NodeInputDescriptor {
  public:
    NodeInputDescriptorBuffer(const std::string& name,
                              const vk::AccessFlags2 access_flags,
                              const vk::PipelineStageFlags2 pipeline_stages,
                              const vk::BufferUsageFlags usage_flags,
                              const uint32_t delay = 0)
        : NodeInputDescriptor(name, access_flags, pipeline_stages, delay),
          usage_flags(usage_flags) {}

    vk::BufferUsageFlags usage_flags;

    static NodeInputDescriptorBuffer compute_read(const std::string& name) {
        return NodeInputDescriptorBuffer{
            name,
            vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::BufferUsageFlagBits::eStorageBuffer,
        };
    }
    static NodeInputDescriptorBuffer transfer_src(const std::string& name) {
        return NodeInputDescriptorBuffer{
            name,
            vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::BufferUsageFlagBits::eTransferSrc,
        };
    }
};

class NodeOutputDescriptor {
  public:
    NodeOutputDescriptor(const std::string& name,
                         const vk::AccessFlags2 access_flags,
                         const vk::PipelineStageFlags2 pipeline_stages,
                         const bool persistent)
        : name(name), access_flags(access_flags), pipeline_stages(pipeline_stages),
          persistent(persistent) {}

    std::string name;
    // The types of access on this output
    vk::AccessFlags2 access_flags;
    // The pipeline stages that access this output
    vk::PipelineStageFlags2 pipeline_stages;
    // Guarantees that the resource stays valid between iterations.
    // Default ist transient (false) -> You can not expect to find what was last written
    bool persistent = false;
};

class NodeOutputDescriptorImage : public NodeOutputDescriptor {
  public:
    NodeOutputDescriptorImage(const std::string& name,
                              const vk::AccessFlags2 access_flags,
                              const vk::PipelineStageFlags2 pipeline_stages,
                              const vk::ImageCreateInfo create_info,
                              const vk::ImageLayout required_layout,
                              const bool persistent = false)
        : NodeOutputDescriptor(name, access_flags, pipeline_stages, persistent),
          create_info(create_info), required_layout(required_layout) {}

    vk::ImageCreateInfo create_info;
    vk::ImageLayout required_layout;

    static NodeOutputDescriptorImage compute_write(const std::string& name,
                                                   const vk::Format format,
                                                   const uint32_t width,
                                                   const uint32_t height,
                                                   const bool persistent = false) {
        const vk::ImageCreateInfo create_info{
            {},
            vk::ImageType::e2D,
            format,
            {width, height, 1},
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

        return NodeOutputDescriptorImage{name,
                                         vk::AccessFlagBits2::eShaderWrite,
                                         vk::PipelineStageFlagBits2::eComputeShader,
                                         create_info,
                                         vk::ImageLayout::eGeneral,
                                         persistent};
    }

    static NodeOutputDescriptorImage transfer_write(const std::string& name,
                                                    const vk::Format format,
                                                    const uint32_t width,
                                                    const uint32_t height,
                                                    const bool persistent = false) {
        const vk::ImageCreateInfo create_info{
            {},
            vk::ImageType::e2D,
            format,
            {width, height, 1},
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

        return NodeOutputDescriptorImage{name,
                                         vk::AccessFlagBits2::eTransferWrite,
                                         vk::PipelineStageFlagBits2::eTransfer,
                                         create_info,
                                         vk::ImageLayout::eTransferDstOptimal,
                                         persistent};
    }
};

class NodeOutputDescriptorBuffer : public NodeOutputDescriptor {
  public:
    NodeOutputDescriptorBuffer(const std::string& name,
                               const vk::AccessFlags2 access_flags,
                               const vk::PipelineStageFlags2 pipeline_stages,
                               const vk::BufferCreateInfo create_info,
                               const bool persistent = false)
        : NodeOutputDescriptor(name, access_flags, pipeline_stages, persistent),
          create_info(create_info) {}

    vk::BufferCreateInfo create_info;
};

} // namespace merian
