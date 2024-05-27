#include "shadertoy.hpp"

#include "merian-nodes/connectors/vk_image_out.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian_nodes {

ShadertoyNode::ShadertoyNode(const SharedContext context,
                             const std::string& path,
                             FileLoader loader,
                             const uint32_t width,
                             const uint32_t height)
    : ComputeNode(context, "ShadertoyNode", sizeof(PushConstant)), width(width), height(height) {
    shader = std::make_shared<ShaderModule>(context, path, loader);
    constant.iResolution = glm::vec2(width, height);
    sw.reset();
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();
}

ShadertoyNode::ShadertoyNode(const SharedContext context,
                             const std::size_t spv_size,
                             const uint32_t spv[],
                             const uint32_t width,
                             const uint32_t height)
    : ComputeNode(context, "ShadertoyNode", sizeof(PushConstant)), width(width), height(height) {
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

std::vector<OutputConnectorHandle>
ShadertoyNode::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
    return {VkImageOut::compute_write("out", vk::Format::eR8G8B8A8Unorm, width, height)};
}

ComputeNode::NodeStatusFlags ShadertoyNode::pre_process([[maybe_unused]] GraphRun& run,
                                                        [[maybe_unused]] const NodeIO& io) {
    NodeStatusFlags flags{};
    if (requires_rebuild) {
        flags |= NodeStatusFlagBits::NEEDS_RECONNECT;
    }
    requires_rebuild = false;
    return flags;
}

SpecializationInfoHandle ShadertoyNode::get_specialization_info() const noexcept {
    return spec_info;
}

const void* ShadertoyNode::get_push_constant([[maybe_unused]] GraphRun& run) {
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

} // namespace merian_nodes
