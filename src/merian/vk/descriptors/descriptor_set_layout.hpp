#pragma once

#include "merian/vk/context.hpp"
#include <spdlog/spdlog.h>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorSetLayout : public std::enable_shared_from_this<DescriptorSetLayout> {

  public:
    DescriptorSetLayout(const SharedContext context,
                        const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                        const vk::DescriptorSetLayoutCreateFlags flags = {})
        : context(context), bindings(bindings) {
        vk::DescriptorSetLayoutCreateInfo info{flags, bindings};
        SPDLOG_DEBUG("create DescriptorSetLayout ({})", fmt::ptr(this));
        layout = context->device.createDescriptorSetLayout(info);
    }

    ~DescriptorSetLayout() {
        SPDLOG_DEBUG("destroy DescriptorSetLayout ({})", fmt::ptr(this));
        context->device.destroyDescriptorSetLayout(layout);
    }

    operator const vk::DescriptorSetLayout&() const {
        return layout;
    }

    const vk::DescriptorSetLayout& get_layout() const {
        return layout;
    }

    const std::vector<vk::DescriptorSetLayoutBinding>& get_bindings() const {
        return bindings;
    }

    const SharedContext& get_context() const {
        return context;
    }

  private:
    const SharedContext context;
    const std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout layout;
};

using DescriptorSetLayoutHandle = std::shared_ptr<DescriptorSetLayout>;

} // namespace merian
