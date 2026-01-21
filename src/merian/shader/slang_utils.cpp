#pragma once

#include "merian/shader/slang_utils.hpp"

namespace merian {

vk::DescriptorType map_slang_to_vk_descriptor_type(slang::BindingType type) {
    switch (type) {
    case slang::BindingType::Texture:
        return vk::DescriptorType::eSampledImage;
    case slang::BindingType::Sampler:
        return vk::DescriptorType::eSampler;
    case slang::BindingType::CombinedTextureSampler:
        return vk::DescriptorType::eCombinedImageSampler;
    case slang::BindingType::ConstantBuffer:
        return vk::DescriptorType::eUniformBuffer;
    case slang::BindingType::RawBuffer:
    case slang::BindingType::MutableRawBuffer:
        return vk::DescriptorType::eStorageBuffer;
    case slang::BindingType::MutableTexture:
        return vk::DescriptorType::eStorageImage;
    case slang::BindingType::RayTracingAccelerationStructure:
        return vk::DescriptorType::eAccelerationStructureKHR;
    case slang::BindingType::InputRenderTarget:
        return vk::DescriptorType::eInputAttachment;
    default:
        SPDLOG_WARN("Unmapped Slang binding type: {}", (int)type);
        return vk::DescriptorType::eUniformBuffer;
    }
}

DescriptorSetLayoutHandle
create_descriptor_set_layout_from_slang_type_layout(const ContextHandle& context,
                                                    slang::TypeLayoutReflection* type_layout) {

    std::vector<vk::DescriptorSetLayoutBinding> bindings;

    // Iterate through all binding ranges
    uint32_t binding_range_count = type_layout->getBindingRangeCount();
    for (uint32_t i = 0; i < binding_range_count; i++) {
        const slang::BindingType kind = type_layout->getBindingRangeType(i);
        const uint32_t count = type_layout->getBindingRangeBindingCount(i);

        // Get the descriptor set and binding indices
        const uint32_t desc_set_index =
            type_layout->getDescriptorSetDescriptorRangeIndexOffset(i, 0);
        const uint32_t binding = desc_set_index; // Simplified - may need adjustment

        const vk::DescriptorType desc_type = map_slang_to_vk_descriptor_type(kind);

        bindings.push_back(vk::DescriptorSetLayoutBinding{binding, desc_type, count,
                                                          vk::ShaderStageFlagBits::eAll, nullptr});
    }

    if (bindings.empty()) {
        SPDLOG_WARN("Created descriptor set layout with no bindings");
    }

    return std::make_shared<DescriptorSetLayout>(context, bindings);
}

BindingInfo get_binding_info_from_offset(const ShaderOffset& offset,
                                         slang::TypeLayoutReflection* type_layout) {
    assert(offset.binding_range_offset < type_layout->getBindingRangeCount());

    slang::BindingType kind = type_layout->getBindingRangeType(offset.binding_range_offset);
    uint32_t count = type_layout->getBindingRangeBindingCount(offset.binding_range_offset);

    // Get binding index
    uint32_t binding =
        type_layout->getDescriptorSetDescriptorRangeIndexOffset(offset.binding_range_offset, 0);

    return BindingInfo{binding, kind, count};
}

} // namespace merian
