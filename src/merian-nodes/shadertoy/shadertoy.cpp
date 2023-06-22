#include "shadertoy.hpp"

namespace merian {

ShadertoyNode::ShadertoyNode(const SharedContext context,
                             const ResourceAllocatorHandle alloc,
                             const std::string& path,
                             FileLoader loader,
                             const uint32_t width,
                             const uint32_t height)
    : ComputeNode(context, alloc, sizeof(PushConstant)), width(width), height(height) {
    shader = std::make_shared<ShaderModule>(context, path, loader);
    constant.iResolution = glm::vec2(width, height);
    sw.reset();
}

ShadertoyNode::ShadertoyNode(const SharedContext context,
                             const ResourceAllocatorHandle alloc,
                             const std::size_t spv_size,
                             const uint32_t spv[],
                             const uint32_t width,
                             const uint32_t height)
    : ComputeNode(context, alloc, sizeof(PushConstant)), width(width), height(height) {
    shader = std::make_shared<ShaderModule>(context, spv_size, spv);
    constant.iResolution = glm::vec2(width, height);
    sw.reset();
}

void ShadertoyNode::set_resolution(uint32_t width, uint32_t height) {
    if (width != this->width || height != this->height) {
        this->width = width;
        this->height = height;
        constant.iResolution = glm::vec2(width, height);
        requires_rebuild = true;
    }
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

void ShadertoyNode::pre_process(NodeStatus& status) {
    status.request_rebuild = requires_rebuild;
    requires_rebuild = false;
}

SpecializationInfoHandle ShadertoyNode::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    return spec_builder.build();
}

const void* ShadertoyNode::get_push_constant() {
    float new_time = sw.seconds();
    constant.iTimeDelta = new_time - constant.iTime;
    constant.iTime = new_time;
    constant.iFrame++;

    return &constant;
}

std::tuple<uint32_t, uint32_t, uint32_t> ShadertoyNode::get_group_count() const noexcept {
    return {(width + local_size_x - 1) / local_size_x, (height + local_size_y - 1) / local_size_y,
            1};
};

ShaderModuleHandle ShadertoyNode::get_shader_module() {
    return shader;
}

} // namespace merian
