#include "node_utils.hpp"

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"

#include <vector>

namespace merian {

// Creates descriptor sets with from the cmd_build inputs.
// An appropriate layout is created if optional_layout is null.
// The graph resources are bound in order input images, input buffers, output images,
// output buffers. The textures contain the images in the same order.
//
// Input images are bound as sampler2d, output images as image2d.
//
// The descriptors for images are created with layout eShaderReadOnly and eGeneral for inputs and
// outputs respectively.
[[nodiscard]] std::tuple<std::vector<TextureHandle>,
                         std::vector<DescriptorSetHandle>,
                         DescriptorPoolHandle,
                         DescriptorSetLayoutHandle>
make_graph_descriptor_sets(const SharedContext context,
                           const ResourceAllocatorHandle allocator,
                           const std::vector<NodeIO>& ios,
                           const DescriptorSetLayoutHandle optional_layout) {
    DescriptorSetLayoutHandle layout;
    if (optional_layout) {
        layout = optional_layout;
    } else {
        auto builder = DescriptorSetLayoutBuilder();
        for (uint32_t i = 0; i < ios[0].image_inputs.size(); i++)
            builder.add_binding_combined_sampler();
        for (uint32_t i = 0; i < ios[0].buffer_inputs.size(); i++)
            builder.add_binding_storage_buffer();

        for (uint32_t i = 0; i < ios[0].image_outputs.size(); i++)
            builder.add_binding_storage_image();
        for (uint32_t i = 0; i < ios[0].buffer_outputs.size(); i++)
            builder.add_binding_storage_buffer();
        layout = builder.build_layout(context);
    }

    std::vector<DescriptorSetHandle> sets;
    std::vector<TextureHandle> textures;
    uint32_t num_sets = ios.size();
    DescriptorPoolHandle pool = std::make_shared<merian::DescriptorPool>(layout, num_sets);

    vk::ImageViewCreateInfo create_image_view{
        {}, VK_NULL_HANDLE, vk::ImageViewType::e2D, {}, {}, first_level_and_layer()};

    TextureHandle out_tex, in_tex;
    for (auto& io : ios) {
        auto set = std::make_shared<merian::DescriptorSet>(pool);
        sets.push_back(set);
        auto update = DescriptorSetUpdate(set);
        uint32_t u_i = 0;

        // in
        for (auto& image_input : io.image_inputs) {
            create_image_view.image = *image_input;
            create_image_view.format = image_input->get_format();
            in_tex = allocator->createTexture(image_input, create_image_view);

            vk::FormatProperties props =
                context->physical_device.physical_device.getFormatProperties(
                    create_image_view.format);
            if (props.optimalTilingFeatures &
                vk::FormatFeatureFlagBits::eSampledImageFilterLinear) {
                in_tex->attach_sampler(allocator->get_sampler_pool()->linear_mirrored_repeat());
            } else {
                in_tex->attach_sampler(allocator->get_sampler_pool()->nearest_mirrored_repeat());
            }
            textures.push_back(in_tex);

            update.write_descriptor_texture(u_i++, in_tex, 0, 1,
                                            vk::ImageLayout::eShaderReadOnlyOptimal);
        }
        for (auto& buffer_input : io.buffer_inputs) {
            update.write_descriptor_buffer(u_i++, buffer_input);
        }

        // out
        for (auto& image_output : io.image_outputs) {
            create_image_view.image = *image_output;
            create_image_view.format = image_output->get_format();
            out_tex = allocator->createTexture(image_output, create_image_view);
            textures.push_back(out_tex);

            update.write_descriptor_texture(u_i++, out_tex, 0, 1, vk::ImageLayout::eGeneral);
        }
        for (auto& buffer_output : io.buffer_outputs) {
            update.write_descriptor_buffer(u_i++, buffer_output);
        }
        update.update(context);
    }

    return std::make_tuple(textures, sets, pool, layout);
}

} // namespace merian
