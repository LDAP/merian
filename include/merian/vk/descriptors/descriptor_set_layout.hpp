#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorSetLayout {

  public:
    DescriptorSetLayout(const vk::DescriptorSetLayout& layout, const std::vector<vk::DescriptorType>&& types)
        : layout(layout), types(std::move(types)) {}

    operator const vk::DescriptorSetLayout&() const {
        return layout;
    }

    const vk::DescriptorSetLayout& get_layout() const {
        return layout;
    }

    const std::vector<vk::DescriptorType>& get_types() const {
        return types;
    }

  private:
    vk::DescriptorSetLayout layout;
    std::vector<vk::DescriptorType> types;
};

} // namespace merian
