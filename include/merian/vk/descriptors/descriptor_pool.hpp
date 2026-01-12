#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <unordered_map>
#include <vulkan/vulkan.hpp>

namespace merian {

class DescriptorSet;
using DescriptorSetHandle = std::shared_ptr<DescriptorSet>;

class DescriptorPool : public std::enable_shared_from_this<DescriptorPool> {

  public:
    virtual ~DescriptorPool() {}

    // Returns the number of sets this pool can allocate for the supplied layout.
    virtual uint32_t can_allocate(const DescriptorSetLayoutHandle& layout) const = 0;

    virtual std::vector<DescriptorSetHandle> allocate(const DescriptorSetLayoutHandle& layout,
                                                      const uint32_t set_count) = 0;

    virtual DescriptorSetHandle allocate(const DescriptorSetLayoutHandle& layout) {
        return allocate(layout, 1).back();
    }
};

using DescriptorPoolHandle = std::shared_ptr<DescriptorPool>;

class VulkanDescriptorPool;
using VulkanDescriptorPoolHandle = std::shared_ptr<VulkanDescriptorPool>;

class VulkanDescriptorPool : public DescriptorPool {

    friend class DescriptorSet;

  public:
    // Allocates one set for each layout
    static std::vector<vk::DescriptorSet>
    allocate_descriptor_sets(const vk::Device& device,
                             const vk::DescriptorPool& pool,
                             const vk::ArrayProxy<vk::DescriptorSetLayout>& layouts) {
        vk::DescriptorSetAllocateInfo info{pool, layouts};
        return device.allocateDescriptorSets(info);
    }

    // Allocates `count` sets for the supplied layout.
    static std::vector<vk::DescriptorSet>
    allocate_descriptor_sets(const vk::Device& device,
                             const vk::DescriptorPool& pool,
                             const vk::DescriptorSetLayout& layout,
                             uint32_t count) {
        std::vector<vk::DescriptorSetLayout> layouts(count, layout);
        return allocate_descriptor_sets(device, pool, layouts);
    }

    static vk::DescriptorSet allocate_descriptor_set(const vk::Device& device,
                                                     const vk::DescriptorPool& pool,
                                                     const vk::DescriptorSetLayout& layout) {
        return allocate_descriptor_sets(device, pool, layout)[0];
    }

    static const inline std::vector<vk::DescriptorPoolSize> DEFAULT_POOL_SIZES = {
        {vk::DescriptorType::eCombinedImageSampler, 65536},
        {vk::DescriptorType::eSampledImage, 16384},
        {vk::DescriptorType::eSampler, 16384},
        {vk::DescriptorType::eUniformBuffer, 16384},
        {vk::DescriptorType::eStorageBuffer, 32768},
        {vk::DescriptorType::eStorageImage, 16384},
        {vk::DescriptorType::eAccelerationStructureKHR, 4096},
        {vk::DescriptorType::eInputAttachment, 4096},
    };
    static const inline uint32_t DEFAULT_POOL_MAX_SETS = 4096;

  private:
    /**
     * @brief      Creates a DescriptorPool that has enough descriptors to allocate set_count
     * DescriptorSets of the supplied DescriptorSetLayouts.
     *
     */
    VulkanDescriptorPool(const DescriptorSetLayoutHandle& layout,
                         const uint32_t set_count = 1,
                         const vk::DescriptorPoolCreateFlags flags = {})
        : merian::VulkanDescriptorPool(layout->get_context(),
                                       layout->get_pool_sizes_as_vector(set_count),
                                       set_count,
                                       flags) {}

    VulkanDescriptorPool(
        const ContextHandle& context,
        const vk::ArrayProxy<vk::DescriptorPoolSize>& pool_sizes = DEFAULT_POOL_SIZES,
        const uint32_t max_sets = DEFAULT_POOL_MAX_SETS,
        const vk::DescriptorPoolCreateFlags flags =
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        : context(context), flags(flags), remaining_set_count(max_sets) {

        remaining_pool_descriptors.reserve(pool_sizes.size());
        for (const auto& size : pool_sizes) {
            remaining_pool_descriptors[size.type] += size.descriptorCount;
        }

        const vk::DescriptorPoolCreateInfo info{flags, max_sets, pool_sizes};
        pool = context->device.createDescriptorPool(info);

        SPDLOG_DEBUG("created DescriptorPool ({})", fmt::ptr(VkDescriptorPool(pool)));
    }

  public:
    static VulkanDescriptorPoolHandle create(const DescriptorSetLayoutHandle& layout,
                                             const uint32_t set_count = 1,
                                             const vk::DescriptorPoolCreateFlags flags = {}) {
        return VulkanDescriptorPoolHandle(new VulkanDescriptorPool(layout, set_count, flags));
    }

    static VulkanDescriptorPoolHandle
    create(const ContextHandle& context,
           const vk::ArrayProxy<vk::DescriptorPoolSize>& pool_sizes = DEFAULT_POOL_SIZES,
           const uint32_t max_sets = DEFAULT_POOL_MAX_SETS,
           const vk::DescriptorPoolCreateFlags flags =
               vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet) {
        return VulkanDescriptorPoolHandle(
            new VulkanDescriptorPool(context, pool_sizes, max_sets, flags));
    }

  public:
    ~VulkanDescriptorPool() {
        SPDLOG_DEBUG("destroy DescriptorPool ({})", fmt::ptr(VkDescriptorPool(pool)));
        context->device.destroyDescriptorPool(pool);
    }

    // -------------------------------------------------------------------------

    uint32_t can_allocate(const DescriptorSetLayoutHandle& layout) const override;

    std::vector<DescriptorSetHandle> allocate(const DescriptorSetLayoutHandle& layout,
                                              const uint32_t set_count) override;

    // -------------------------------------------------------------------------

    const std::unordered_map<vk::DescriptorType, uint32_t>& get_allocated_descriptor_count() const {
        return allocated_pool_descriptors;
    }

    const uint32_t& get_allocated_set_count() const {
        return allocated_set_count;
    }

    operator const vk::DescriptorPool&() const {
        return pool;
    }

    const vk::DescriptorPool& get_pool() const {
        return pool;
    }

    const ContextHandle& get_context() const {
        return context;
    }

    vk::DescriptorPoolCreateFlags get_create_flags() {
        return flags;
    }

    bool supports_free_descriptor_set() const {
        return bool(flags & vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    }

  private:
    void free(const DescriptorSet* set);

  private:
    const ContextHandle context;
    const vk::DescriptorPoolCreateFlags flags;

    std::unordered_map<vk::DescriptorType, uint32_t> remaining_pool_descriptors;
    std::unordered_map<vk::DescriptorType, uint32_t> allocated_pool_descriptors;
    uint32_t remaining_set_count = 0;
    uint32_t allocated_set_count = 0;

    vk::DescriptorPool pool;
};

class ResizingVulkanDescriptorPool;
using ResizingDescriptorPoolHandle = std::shared_ptr<ResizingVulkanDescriptorPool>;

class ResizingVulkanDescriptorPool : public DescriptorPool {
  private:
    ResizingVulkanDescriptorPool(const ContextHandle& context) : context(context) {
        pools.reserve(16);
        pools.emplace_back(VulkanDescriptorPool::create(context));
    }

  public:
    static ResizingDescriptorPoolHandle create(const ContextHandle& context) {
        return ResizingDescriptorPoolHandle(new ResizingVulkanDescriptorPool(context));
    }

  public:
    // Returns the number of sets this pool can allocate for the supplied layout.
    uint32_t can_allocate(const DescriptorSetLayoutHandle& /*layout*/) const override {
        // if we need we create a pool that can allocate whatever is needed.
        return uint32_t(-1);
    }

    std::vector<DescriptorSetHandle> allocate(const DescriptorSetLayoutHandle& layout,
                                              const uint32_t set_count) override;

  private:
    const ContextHandle context;

    std::vector<VulkanDescriptorPoolHandle> pools;
};

} // namespace merian
