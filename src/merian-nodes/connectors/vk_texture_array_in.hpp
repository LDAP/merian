#pragma once

#include "vk_texture_array_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

namespace merian_nodes {

class TextureArrayIn;
using TextureArrayInHandle = std::shared_ptr<TextureArrayIn>;

class TextureArrayIn
    : public TypedInputConnector<TextureArrayOutHandle, const TextureArrayResource&> {
    friend class TextureArrayOut;

  public:
    // A descriptor binding is only created if stage flags are supplied.
    TextureArrayIn(const std::string& name,
                   const std::optional<vk::ShaderStageFlags>& stage_flags = {});

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               GraphResourceHandle& resource,
                               DescriptorSetUpdate& update) override;

    const TextureArrayResource& resource(const GraphResourceHandle& resource) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

    static TextureArrayInHandle create(const std::string& name,
                                       const std::optional<vk::ShaderStageFlags>& stage_flags);

  private:
    const std::optional<vk::ShaderStageFlags> stage_flags = {};
    // set from output in create resource
    uint32_t array_size;
};

} // namespace merian_nodes
