#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/sampler/sampler.hpp"
#include "merian/utils/hash.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <unordered_map>

namespace merian {

/**
 * @brief      This class describes a sampler pool.
 * 
 * Holds weak references to sampler, to return the same sampler,
 * if an identical configuration is requested.
 */
class SamplerPool : public std::enable_shared_from_this<SamplerPool> {
  public:
    SamplerPool(SamplerPool const&) = delete;
    SamplerPool& operator=(SamplerPool const&) = delete;
    SamplerPool() = delete;

    SamplerPool(const SharedContext& context) : context(context) {
        SPDLOG_DEBUG("create sampler pool ({})", fmt::ptr(this));
    }
    ~SamplerPool();

    /* creates a new sampler or re-uses an existing one.
     * createInfo may contain VkSamplerReductionModeCreateInfo and VkSamplerYcbcrConversionCreateInfo
     */
    SamplerHandle acquire_sampler(const vk::SamplerCreateInfo& createInfo);


    SamplerHandle linear_mirrored_repeat();

    SamplerHandle nearest_mirrored_repeat();

    SamplerHandle linear_repeat();

    SamplerHandle nearest_repeat();

  private:
    struct SamplerState {
        vk::SamplerCreateInfo createInfo{};
        vk::SamplerReductionModeCreateInfo reduction{};
        vk::SamplerYcbcrConversionCreateInfo ycbr{};

        SamplerState() {}

        auto operator<=>(const SamplerState&) const = default;
    };

    struct Chain {
        VkStructureType sType;
        const Chain* pNext;
    };

    const SharedContext context;
    
    std::unordered_map<SamplerState, std::weak_ptr<Sampler>, HashAligned32<SamplerState>> state_map;
};

using SamplerPoolHandle = std::shared_ptr<SamplerPool>;

} // namespace merian
