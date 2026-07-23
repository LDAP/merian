#include "merian-graph/connectors/image/vk_image_in_storage.hpp"

namespace merian {

void VkStorageImageIn::bind(ShaderCursor& cursor,
                            const GraphResourceHandle& resource,
                            const ResourceAllocatorHandle& allocator,
                            [[maybe_unused]] const ConnectorAccess& access) {
    const auto write = [&](ShaderCursor field, const uint32_t index) {
        const TextureHandle tex =
            resource ? debugable_ptr_cast<ImageArrayResource>(resource)->get_texture(index)
                     : nullptr;
        field.write(tex ? tex->get_view() : allocator->get_dummy_storage_image_view(),
                    vk::ImageLayout::eGeneral);
    };
    if (get_array_size() == 1) {
        write(cursor, 0);
    } else {
        for (uint32_t i = 0; i < get_array_size(); i++) {
            write(cursor[i], i);
        }
    }
}

VkStorageImageInHandle VkStorageImageIn::create() {
    return std::make_shared<VkStorageImageIn>();
}

} // namespace merian
