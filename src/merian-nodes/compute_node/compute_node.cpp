#pragma once

#include "compute_node.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

ComputeNode::ComputeNode(const SharedContext context,
                         const ResourceAllocatorHandle allocator,
                         const std::optional<uint32_t> push_constant_size)
    : context(context), allocator(allocator),
      push_constant_size(push_constant_size) {}

void ComputeNode::cmd_build(const vk::CommandBuffer&,
                            const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                            const std::vector<std::vector<merian::BufferHandle>>& buffer_inputs,
                            const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                            const std::vector<std::vector<merian::BufferHandle>>& buffer_outputs) {
    if (!pipe) {
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

        auto pipe_builder = PipelineLayoutBuilder(context);
        if (push_constant_size.has_value()) {
            pipe_builder.add_push_constant(push_constant_size.value());
        }
        auto pipe_layout = pipe_builder.add_descriptor_set_layout(layout).build_pipeline_layout();
        pipe = std::make_shared<ComputePipeline>(pipe_layout, get_shader_module(), get_specialization_info());
    }

    sets.clear();
    in_textures.clear();
    out_textures.clear();
    uint32_t num_sets = image_outputs.size();
    pool = std::make_shared<merian::DescriptorPool>(layout, num_sets);

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
            out_textures.push_back(in_tex);

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
            out_textures.push_back(out_tex);

            update.write_descriptor_texture(u_i++, out_tex);
        }
        for (auto& buffer_output : buffer_outputs[i]) {
            update.write_descriptor_buffer(u_i++, buffer_output);
        }

        update.update(context);
    }
}

void ComputeNode::cmd_process(const vk::CommandBuffer& cmd,
                              const uint64_t,
                              const uint32_t set_index,
                              const std::vector<ImageHandle>&,
                              const std::vector<BufferHandle>&,
                              const std::vector<ImageHandle>&,
                              const std::vector<BufferHandle>&) {
    pipe->bind(cmd);
    pipe->bind_descriptor_set(cmd, sets[set_index]);
    if (push_constant_size.has_value())
        pipe->push_constant(cmd, get_push_constant());
    auto [x, y, z] = get_group_count();
    cmd.dispatch(x, y, z);
}

} // namespace merian
