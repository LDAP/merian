#pragma once

#include <memory>
#include <optional>

#include <vulkan/vulkan.hpp>

namespace merian {

class Connector : public std::enable_shared_from_this<Connector> {
  public:
    Connector(const std::string& name) : name(name) {}

    virtual ~Connector() = 0;

    // If the resource of this connector should be available in a shader
    // return the vk::ShaderStageFlags, vk::DescriptorType and the descriptor count here.
    virtual std::optional<std::tuple<vk::ShaderStageFlags, vk::DescriptorType, uint32_t>>
    get_descriptor_info() const {
        return std::nullopt;
    }

    
    
  public:
    const std::string name;
};

} // namespace merian
