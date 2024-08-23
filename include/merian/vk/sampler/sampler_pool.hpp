#pragma once

#include "merian/utils/hash.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/sampler/sampler.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_hash.hpp>

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

    SamplerPool(const ContextHandle& context) : context(context) {
        SPDLOG_DEBUG("create sampler pool ({})", fmt::ptr(this));
    }
    ~SamplerPool();

    /* creates a new sampler or re-uses an existing one.
     * createInfo may contain VkSamplerReductionModeCreateInfo and
     * VkSamplerYcbcrConversionCreateInfo
     */
    SamplerHandle acquire_sampler(const vk::SamplerCreateInfo& createInfo);

    SamplerHandle for_filter_and_address_mode(
        const vk::Filter mag_filter = vk::Filter::eNearest,
        const vk::Filter min_filter = vk::Filter::eLinear,
        const vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eRepeat,
        const bool anisotropy = true);

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

    struct SamplerStateHash {
        std::size_t operator()(const SamplerState& s) const {
            return hash_val(create_hash(s.createInfo), reduction_hash(s.reduction),
                            ycbcr_conversion_hash(s.ycbr));
        }

      private:
        std::hash<vk::SamplerCreateInfo> create_hash;
        std::hash<vk::SamplerReductionModeCreateInfo> reduction_hash;
        std::hash<vk::SamplerYcbcrConversionCreateInfo> ycbcr_conversion_hash;
    };

    struct Chain {
        VkStructureType sType;
        const Chain* pNext;
    };

    const ContextHandle context;

    std::unordered_map<SamplerState, std::weak_ptr<Sampler>, SamplerStateHash> state_map;
};

using SamplerPoolHandle = std::shared_ptr<SamplerPool>;

} // namespace merian
