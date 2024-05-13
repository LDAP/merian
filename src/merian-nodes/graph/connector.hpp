#pragma once

#include <memory>

#include "graph_run.hpp"
#include "resource.hpp"

#include <vulkan/vulkan.hpp>

namespace merian_nodes {

class Connector : public std::enable_shared_from_this<Connector> {
  public:
    Connector(const std::string& name) : name(name) {}

    virtual ~Connector() = 0;

    // If the resource should be available in a shader, return the vk::ShaderStageFlags,
    // vk::DescriptorType and the descriptor count here.
    virtual std::optional<std::tuple<vk::ShaderStageFlags, vk::DescriptorType, uint32_t>>
    get_descriptor_info() const {
        return std::nullopt;
    }

    // Called right after the node with this connector has finished node.pre_process() and before
    // node.process(). This is the place to insert barriers, if necessary. Also, you can validate
    // here that the node did use the output correctly (set the resource in pre_process for example)
    // and throw merian::graph_errors::connector_error if not.
    //
    // The graph supplies here the resource for the current iteration (depending on delay and such).
    virtual void on_pre_process([[maybe_unused]] GraphRun& run,
                                [[maybe_unused]] const vk::CommandBuffer& cmd,
                                [[maybe_unused]] GraphResourceHandle& resource) {}

    // Called right after the node with this connector has finished node.process(). For example, you
    // can validate here that the node did use the output correctly (set the resource for example)
    // and throw merian::graph_errors::connector_error if not.
    //
    // The graph supplies here the resource for the current iteration (depending on delay and such).
    virtual void on_post_process([[maybe_unused]] GraphRun& run,
                                 [[maybe_unused]] const vk::CommandBuffer& cmd,
                                 [[maybe_unused]] GraphResourceHandle& resource) {}

  public:
    const std::string name;
};

using ConnectorHandle = std::shared_ptr<Connector>;

} // namespace merian_nodes
