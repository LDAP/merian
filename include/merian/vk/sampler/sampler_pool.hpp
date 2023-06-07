#pragma once

#include "merian/utils/vector_utils.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/sampler/sampler.hpp"

#include <cfloat>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <functional>
#include <unordered_map>
#include <vector>

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
    SamplerHandle acquireSampler(const vk::SamplerCreateInfo& createInfo);

  private:
    struct SamplerState {
        vk::SamplerCreateInfo createInfo;
        vk::SamplerReductionModeCreateInfo reduction;
        vk::SamplerYcbcrConversionCreateInfo ycbr;

        SamplerState() {}
        bool operator==(const SamplerState& other) const = default;
    };

    struct Chain {
        VkStructureType sType;
        const Chain* pNext;
    };

    struct Entry {
        std::weak_ptr<Sampler> sampler;
        uint32_t nextFreeIndex = ~0;
        SamplerState state;
    };

    const SharedContext context;
    uint32_t freeIndex = ~0;

    std::vector<Entry> entries;
    std::unordered_map<SamplerState, uint32_t, HashAligned32<SamplerState>> state_map;
};

} // namespace merian
