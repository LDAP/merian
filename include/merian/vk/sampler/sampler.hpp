#pragma once

#include "merian/vk/context.hpp"

#include <memory>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace merian {

/**
 * @brief      This class describes a vk::Sampler with automatic cleanup.
 */
class Sampler : public std::enable_shared_from_this<Sampler>, public Object {
public:

    Sampler(const ContextHandle& context, const vk::SamplerCreateInfo& create_info) : context(context) {
        SPDLOG_DEBUG("create sampler ({})", fmt::ptr(this));
        sampler = context->device.createSampler(create_info);
    }

    ~Sampler() {
        SPDLOG_DEBUG("destroy sampler ({})", fmt::ptr(this));
        context->device.destroySampler(sampler);
    }

    operator const vk::Sampler&() const {
        return sampler;
    }

    const vk::Sampler& operator*() {
        return sampler;
    }

    const vk::Sampler& get_sampler() const {
        return sampler;
    }

private:
    const ContextHandle context;
    vk::Sampler sampler;
};

using SamplerHandle = std::shared_ptr<Sampler>;
}
