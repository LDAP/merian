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
                        const uint32_t delay);

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
                             const uint32_t delay = 0);

    vk::ImageLayout required_layout;
    vk::ImageUsageFlags usage_flags;

    static NodeInputDescriptorImage compute_read(const std::string& name, const uint32_t delay = 0);

    static NodeInputDescriptorImage transfer_src(const std::string& name, const uint32_t delay = 0);
};

struct NodeInputDescriptorBuffer : public NodeInputDescriptor {
  public:
    NodeInputDescriptorBuffer(const std::string& name,
                              const vk::AccessFlags2 access_flags,
                              const vk::PipelineStageFlags2 pipeline_stages,
                              const vk::BufferUsageFlags usage_flags,
                              const uint32_t delay = 0);

    vk::BufferUsageFlags usage_flags;

    static NodeInputDescriptorBuffer compute_read(const std::string& name);

    static NodeInputDescriptorBuffer transfer_src(const std::string& name);
};

class NodeOutputDescriptor {
  public:
    NodeOutputDescriptor(const std::string& name,
                         const vk::AccessFlags2 access_flags,
                         const vk::PipelineStageFlags2 pipeline_stages,
                         const bool persistent);

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
                              const bool persistent = false);

    vk::ImageCreateInfo create_info;
    vk::ImageLayout required_layout;

    static NodeOutputDescriptorImage compute_write(const std::string& name,
                                                   const vk::Format format,
                                                   const vk::Extent3D extent,
                                                   const bool persistent = false);

    static NodeOutputDescriptorImage compute_write(const std::string& name,
                                                   const vk::Format format,
                                                   const uint32_t width,
                                                   const uint32_t height,
                                                   const bool persistent = false);

    static NodeOutputDescriptorImage compute_read_write(const std::string& name,
                                                        const vk::Format format,
                                                        const vk::Extent3D extent,
                                                        const bool persistent = false);

    static NodeOutputDescriptorImage transfer_write(const std::string& name,
                                                    const vk::Format format,
                                                    const vk::Extent3D extent,
                                                    const bool persistent = false);

    static NodeOutputDescriptorImage transfer_write(const std::string& name,
                                                    const vk::Format format,
                                                    const uint32_t width,
                                                    const uint32_t height,
                                                    const bool persistent = false);
};

class NodeOutputDescriptorBuffer : public NodeOutputDescriptor {
  public:
    NodeOutputDescriptorBuffer(const std::string& name,
                               const vk::AccessFlags2 access_flags,
                               const vk::PipelineStageFlags2 pipeline_stages,
                               const vk::BufferCreateInfo create_info,
                               const bool persistent = false);

    vk::BufferCreateInfo create_info;
};

} // namespace merian
