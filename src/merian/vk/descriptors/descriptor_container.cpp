#include "merian/vk/descriptors/descriptor_container.hpp"

namespace merian {

DescriptorContainer::~DescriptorContainer() {}

void DescriptorContainer::replay_to(DescriptorContainer& target) const {
    assert(layout == target.get_layout());

    for (uint32_t b = 0; b < layout->get_bindings().size(); b++) {
        const auto& binding = layout->get_bindings()[b];
        const uint32_t binding_number = binding.binding;

        for (uint32_t a = 0; a < binding.descriptorCount; a++) {
            const uint32_t index = layout->get_binding_offset(binding_number, a);
            const auto& resource = resources[index];
            if (!resource) {
                continue;
            }

            const auto& info = infos[index];
            std::visit(
                [&](const auto& info_value) {
                    using T = std::decay_t<decltype(info_value)>;
                    if constexpr (std::is_same_v<T, vk::DescriptorBufferInfo>) {
                        auto buffer = std::static_pointer_cast<Buffer>(resource);
                        target.queue_descriptor_write_buffer(
                            binding_number, buffer, info_value.offset, info_value.range, a);
                    } else if constexpr (std::is_same_v<T, vk::DescriptorImageInfo>) {
                        const vk::DescriptorType desc_type =
                            layout->get_type_for_binding(binding_number);
                        if (desc_type == vk::DescriptorType::eSampler) {
                            auto sampler = std::static_pointer_cast<Sampler>(resource);
                            target.queue_descriptor_write_sampler(binding_number, sampler, a);
                        } else if (desc_type == vk::DescriptorType::eCombinedImageSampler) {
                            auto texture = std::static_pointer_cast<Texture>(resource);
                            target.queue_descriptor_write_texture(binding_number, texture, a,
                                                                  info_value.imageLayout);
                        } else {
                            // SampledImage or StorageImage
                            auto image_view = std::static_pointer_cast<ImageView>(resource);
                            target.queue_descriptor_write_image(binding_number, image_view, a,
                                                                info_value.imageLayout);
                        }
                    } else if constexpr (std::is_same_v<
                                             T, vk::WriteDescriptorSetAccelerationStructureKHR>) {
                        auto as = std::static_pointer_cast<AccelerationStructure>(resource);
                        target.queue_descriptor_write_acceleration_structure(binding_number, as, a);
                    }
                },
                info);
        }
    }
}

} // namespace merian
