#include "node_utils.hpp"

#include "vk/descriptors/descriptor_set_layout_builder.hpp"
#include "vk/descriptors/descriptor_set_update.hpp"

#include <vector>

namespace merian {

// Creates descriptor sets with from the cmd_build inputs.
// An appropriate layout is created if optional_layout is null.
// The graph resources are bound in order input images, input buffers, output images,
// output buffers.
// Input images are bound as sampler2d, output images as image2d.
[[nodiscard]]
std::tuple<std::vector<TextureHandle>,
           std::vector<DescriptorSetHandle>,
           DescriptorPoolHandle,
           DescriptorSetLayoutHandle>
make_descriptor_sets(const SharedContext context,
                     const ResourceAllocatorHandle allocator,
                     const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                     const std::vector<std::vector<merian::BufferHandle>>& buffer_inputs,
                     const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                     const std::vector<std::vector<merian::BufferHandle>>& buffer_outputs,
                     const DescriptorSetLayoutHandle optional_layout) {
    DescriptorSetLayoutHandle layout;
    if (optional_layout) {
        layout = optional_layout;
    } else {
        auto builder = DescriptorSetLayoutBuilder();
        for (uint32_t i = 0; i < image_inputs[0].size(); i++)
            builder.add_binding_combined_sampler();
        for (uint32_t i = 0; i < buffer_inputs[0].size(); i++)
            builder.add_binding_storage_buffer();

        for (uint32_t i = 0; i < image_outputs[0].size(); i++)
            builder.add_binding_storage_image();
        for (uint32_t i = 0; i < buffer_outputs[0].size(); i++)
            builder.add_binding_storage_buffer();
        layout = builder.build_layout(context);
    }

    std::vector<DescriptorSetHandle> sets;
    std::vector<TextureHandle> textures;
    uint32_t num_sets = image_outputs.size();
    DescriptorPoolHandle pool = std::make_shared<merian::DescriptorPool>(layout, num_sets);

    vk::ImageViewCreateInfo create_image_view{
        {}, VK_NULL_HANDLE, vk::ImageViewType::e2D, {}, {}, first_level_and_layer()};

    TextureHandle out_tex, in_tex;
    for (uint32_t i = 0; i < num_sets; i++) {
        auto set = std::make_shared<merian::DescriptorSet>(pool);
        sets.push_back(set);
        auto update = DescriptorSetUpdate(set);
        uint32_t u_i = 0;

        // in
        for (auto& image_input : image_inputs[i]) {
            create_image_view.image = *image_input;
            create_image_view.format = image_input->get_format();
            in_tex = allocator->createTexture(image_input, create_image_view);
            in_tex->attach_sampler(allocator->get_sampler_pool()->linear_mirrored_repeat());
            textures.push_back(in_tex);

            update.write_descriptor_texture(u_i++, in_tex);
        }
        for (auto& buffer_input : buffer_inputs[i]) {
            update.write_descriptor_buffer(u_i++, buffer_input);
        }

        // out
        for (auto& image_output : image_outputs[i]) {
            create_image_view.image = *image_output;
            create_image_view.format = image_output->get_format();
            out_tex = allocator->createTexture(image_output, create_image_view);
            textures.push_back(out_tex);

            update.write_descriptor_texture(u_i++, out_tex);
        }
        for (auto& buffer_output : buffer_outputs[i]) {
            update.write_descriptor_buffer(u_i++, buffer_output);
        }
        update.update(context);
    }

    return std::make_tuple(textures, sets, pool, layout);
}

} // namespace merian
