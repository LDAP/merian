#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

// Allocates one set for each layout
inline std::vector<vk::DescriptorSet>
allocate_descriptor_sets(const vk::Device& device, const vk::DescriptorPool& pool, const std::vector<vk::DescriptorSetLayout>& layouts) {
    vk::DescriptorSetAllocateInfo info{pool, layouts};
    return device.allocateDescriptorSets(info);
}

// Allocates `count` sets for the supplied layout.
inline std::vector<vk::DescriptorSet>
allocate_descriptor_sets(const vk::Device& device, const vk::DescriptorPool& pool, const vk::DescriptorSetLayout& layout, uint32_t count) {
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);
    return allocate_descriptor_sets(device, pool, layouts);
}

inline vk::DescriptorSet
allocate_descriptor_set(const vk::Device& device, const vk::DescriptorPool& pool, const vk::DescriptorSetLayout& layout) {
    return allocate_descriptor_sets(device, pool, {layout})[0];
}

// A DescriptorSet that knows its layout -> Can be used to simplify DescriptorSetUpdate.
class DescriptorSet {

  public:
    DescriptorSet(const vk::Device& device, const vk::DescriptorPool& pool, const std::shared_ptr<DescriptorSetLayout> layout) {
        set = allocate_descriptor_set(device, pool, *layout.get());
        this->layout = layout;
    }


    operator const vk::DescriptorSet&() const {
        return set;
    }

    operator const vk::DescriptorSetLayout&() const {
        return *layout.get();
    }


    const vk::DescriptorSet& get_set() const {
        return set;
    }

    const DescriptorSetLayout& get_layout() const {
        return *layout.get();
    }

    vk::DescriptorType get_type_for_binding(uint32_t binding) const {
        return layout->get_types()[binding];
    }

  private:
    vk::DescriptorSet set;
    std::shared_ptr<DescriptorSetLayout> layout;
};

} // namespace merian
