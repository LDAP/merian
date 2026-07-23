#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"

namespace merian {

VkSampledImageIn::VkSampledImageIn(const std::optional<SamplerHandle>& overwrite_sampler)
    : overwrite_sampler(overwrite_sampler) {}

void VkSampledImageIn::bind(ShaderCursor& cursor,
                            const GraphResourceHandle& resource,
                            const ResourceAllocatorHandle& allocator,
                            [[maybe_unused]] const ConnectorAccess& access) {
    const auto write = [&](ShaderCursor field, const uint32_t index) {
        TextureHandle tex =
            resource ? debugable_ptr_cast<ImageArrayResource>(resource)->get_texture(index)
                     : nullptr;
        if (tex && overwrite_sampler) {
            tex = Texture::create(tex->get_view(), *overwrite_sampler);
        }
        field.write(tex ? tex : allocator->get_dummy_texture(), vk::ImageLayout::eGeneral);
    };
    if (get_array_size() == 1) {
        write(cursor, 0);
    } else {
        for (uint32_t i = 0; i < get_array_size(); i++) {
            write(cursor[i], i);
        }
    }
}

VkSampledImageInHandle
VkSampledImageIn::create(const std::optional<SamplerHandle>& overwrite_sampler) {
    return std::make_shared<VkSampledImageIn>(overwrite_sampler);
}

} // namespace merian
