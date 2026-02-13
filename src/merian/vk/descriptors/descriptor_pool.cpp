#include "merian/vk/descriptors/descriptor_pool.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"

namespace merian {

uint32_t DescriptorPool::can_allocate(const DescriptorSetLayoutHandle& layout) const {
    uint32_t max_sets = remaining_set_count;
    for (const auto& required_pool_size : layout->get_pool_sizes()) {
        const auto it = remaining_pool_descriptors.find(required_pool_size.first);
        const uint32_t remaining_descriptor_count =
            (it != remaining_pool_descriptors.end()) ? it->second : 0;
        max_sets = std::min(max_sets, remaining_descriptor_count / required_pool_size.second);
    }
    return max_sets;
}

std::vector<DescriptorSetHandle>
DescriptorPool::allocate(const DescriptorSetLayoutHandle& layout, const uint32_t set_count) {
    assert(remaining_set_count >= set_count && "out of descriptor sets");
    assert(set_count > 0);
    assert(layout->supports_descriptor_set());

    remaining_set_count -= set_count;
    allocated_set_count += set_count;
    for (const auto& binding : layout->get_bindings()) {
        assert(remaining_pool_descriptors[binding.descriptorType] >=
                   binding.descriptorCount * set_count &&
               "out of descriptors");
        remaining_pool_descriptors[binding.descriptorType] -= binding.descriptorCount * set_count;
        allocated_pool_descriptors[binding.descriptorType] += binding.descriptorCount * set_count;
    }

    const std::vector<vk::DescriptorSet> allocated_sets =
        allocate_descriptor_sets(context->get_device()->get_device(), pool, *layout, set_count);

    std::vector<DescriptorSetHandle> sets(allocated_sets.size());

    const DescriptorPoolHandle pool_ptr =
        static_pointer_cast<DescriptorPool>(shared_from_this());

    for (uint32_t i = 0; i < allocated_sets.size(); i++) {
        sets[i] = DescriptorSet::create(pool_ptr, layout, allocated_sets[i]);
    }

    return sets;
}

void DescriptorPool::free(const DescriptorSet* set) {
    remaining_set_count++;
    allocated_set_count--;
    for (const auto& size : set->get_layout()->get_pool_sizes()) {
        remaining_pool_descriptors[size.first] += size.second;
        allocated_pool_descriptors[size.first] -= size.second;
    }

    if (supports_free_descriptor_set()) {
        // DescriptorSet can be given back to the DescriptorPool
        SPDLOG_DEBUG("freeing DescriptorSet ({})",
                     fmt::ptr(static_cast<VkDescriptorSet>(set->set)));
        context->get_device()->get_device().freeDescriptorSets(pool, set->set);
    } else {
        SPDLOG_DEBUG("destroying DescriptorSet ({}) but not freeing since the pool was not "
                     "created with the {} bit set.",
                     fmt::ptr(static_cast<VkDescriptorSet>(set->set)),
                     vk::to_string(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));
    }
}

// -----------------------------------------------------

namespace {

uint32_t allocate_from_pool(const DescriptorSetAllocatorHandle& pool,
                            const DescriptorSetLayoutHandle& layout,
                            std::vector<DescriptorSetHandle>& insert_into,
                            const uint32_t max_count) {
    const uint32_t set_count = std::min(max_count, pool->can_allocate(layout));
    if (set_count > 0) {
        std::vector<DescriptorSetHandle> allocated = pool->allocate(layout, set_count);
        merian::insert_all(insert_into, allocated);
    }
    return set_count;
}

} // namespace

std::vector<DescriptorSetHandle>
ResizingDescriptorPool::allocate(const DescriptorSetLayoutHandle& layout,
                                       const uint32_t set_count) {

    std::vector<DescriptorSetHandle> result;
    result.reserve(set_count);

    uint32_t remaining_count = set_count;
    for (auto pool_it = pools.rbegin(); pool_it != pools.rend(); pool_it++) {
        // we add new pools at the back
        remaining_count -= allocate_from_pool(*pool_it, layout, result, remaining_count);

        if (*pool_it != pools.back())
            std::swap(*pool_it, pools.back()); // LRU
    }

    if (remaining_count > 0) {
        // -> allocate new pool

        // Count how much is needed and add double the amount of what was allocated until now
        auto new_pool_sizes = layout->get_pool_sizes();
        uint32_t new_pool_set_count = set_count;
        for (const auto& pool : pools) {
            for (const auto& size : pool->get_allocated_descriptor_count()) {
                new_pool_sizes[size.first] += 2 * size.second;
            }
            new_pool_set_count += 2 * pool->get_allocated_set_count();
        }

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
        std::vector<std::string> debug_sizes;
        debug_sizes.reserve(new_pool_sizes.size());
        for (const auto& s : new_pool_sizes) {
            debug_sizes.emplace_back(fmt::format("({}, {})", vk::to_string(s.first), s.second));
        }
        SPDLOG_DEBUG("allocating new DescriptorPool for {} descriptor sets with pool sizes:\n{}",
                     new_pool_set_count, fmt::join(debug_sizes, "\n"));
#endif

        pools.emplace_back(DescriptorPool::create(
            context, DescriptorSetLayout::pool_sizes_to_vector(new_pool_sizes, 1),
            new_pool_set_count));

        remaining_count -= allocate_from_pool(pools.back(), layout, result, remaining_count);
    }

    assert(result.size() == set_count);
    assert(remaining_count == 0);
    return result;
}

} // namespace merian
