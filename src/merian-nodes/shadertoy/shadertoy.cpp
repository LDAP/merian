#pragma once

#include "shadertoy.hpp"

namespace merian {

ShadertoyNode::ShadertoyNode(const SharedContext context,
                             const ResourceAllocatorHandle alloc,
                             std::string path,
                             FileLoader loader,
                             uint32_t width,
                             uint32_t height)
    : context(context), alloc(alloc), width(width), height(height) {
    auto builder = DescriptorSetLayoutBuilder().add_binding_storage_image(); // result
    layout = builder.build_layout(context);

    auto shader = std::make_shared<ShaderModule>(context, path, loader);
    auto pipe_layout = PipelineLayoutBuilder(context)
                           .add_descriptor_set_layout(layout)
                           .add_push_constant<PushConstant>(vk::ShaderStageFlagBits::eCompute)
                           .build_pipeline_layout();
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    auto spec_info = spec_builder.build();
    pipe = std::make_shared<ComputePipeline>(pipe_layout, shader, spec_info);

    sw.reset();
}

void ShadertoyNode::set_resolution(uint32_t width, uint32_t height) {
    if (width != this->width || height != this->height) {
        this->width = width;
        this->height = height;
        requires_rebuild = true;
    }
}

void ShadertoyNode::pre_process(NodeStatus& status) {
    status.request_rebuild = requires_rebuild;
}

std::tuple<std::vector<merian::NodeInputDescriptorImage>,
           std::vector<merian::NodeInputDescriptorBuffer>>
ShadertoyNode::describe_inputs() {
    return {};
}

std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
           std::vector<merian::NodeOutputDescriptorBuffer>>
ShadertoyNode::describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>&,
                                const std::vector<merian::NodeOutputDescriptorBuffer>&) {

    return {
        {merian::NodeOutputDescriptorImage::compute_write("result", vk::Format::eR8G8B8A8Unorm,
                                                          width, height)},
        {},
    };
}

void ShadertoyNode::cmd_build(const vk::CommandBuffer&,
                              const std::vector<std::vector<merian::ImageHandle>>&,
                              const std::vector<std::vector<merian::BufferHandle>>&,
                              const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                              const std::vector<std::vector<merian::BufferHandle>>&) {
    sets.clear();
    textures.clear();

    uint32_t num_sets = image_outputs.size();

    pool = std::make_shared<merian::DescriptorPool>(layout, num_sets);
    vk::ImageViewCreateInfo create_image_view{
        {}, VK_NULL_HANDLE,         vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm,
        {}, first_level_and_layer()};

    for (uint32_t i = 0; i < num_sets; i++) {
        auto set = std::make_shared<merian::DescriptorSet>(pool);
        create_image_view.image = *image_outputs[i][0];
        auto tex = alloc->createTexture(image_outputs[i][0], create_image_view);
        DescriptorSetUpdate(set).write_descriptor_texture(0, tex).update(context);
        sets.push_back(set);
        textures.push_back(tex);
    }
    constant.iResolution = glm::vec2(width, height);
    requires_rebuild = false;
}

void ShadertoyNode::cmd_process(const vk::CommandBuffer& cmd,
                                const uint64_t iteration,
                                const uint32_t set_index,
                                const std::vector<merian::ImageHandle>&,
                                const std::vector<merian::BufferHandle>&,
                                const std::vector<merian::ImageHandle>&,
                                const std::vector<merian::BufferHandle>&) {
    float new_time = sw.seconds();
    constant.iTimeDelta = new_time - constant.iTime;
    constant.iTime = new_time;
    constant.iFrame = iteration;

    pipe->bind(cmd);
    pipe->bind_descriptor_set(cmd, sets[set_index]);
    pipe->push_constant<PushConstant>(cmd, constant);
    cmd.dispatch((width + local_size_x - 1) / local_size_x,
                 (height + local_size_y - 1) / local_size_y, 1);
}

} // namespace merian
