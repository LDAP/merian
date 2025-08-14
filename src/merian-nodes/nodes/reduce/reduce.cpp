#include "merian-nodes/nodes/reduce/reduce.hpp"

#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/math.hpp"

#include <regex>

namespace merian_nodes {

Reduce::Reduce(const ContextHandle& context, const std::optional<vk::Format>& output_format)
    : AbstractCompute(context), output_format(output_format) {}

Reduce::~Reduce() {}

std::vector<InputConnectorHandle> Reduce::describe_inputs() {
    if (input_connectors.size() != number_inputs) {
        input_connectors.clear();
        for (uint32_t i = 0; i < number_inputs; i++) {
            input_connectors.emplace_back(
                VkSampledImageIn::compute_read(fmt::format("input_{}", i), 0, true));
        }
    }

    return {input_connectors.begin(), input_connectors.end()};
}

std::vector<OutputConnectorHandle> Reduce::describe_outputs(const NodeIOLayout& io_layout) {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);

    vk::Format format = output_format.value_or(vk::Format::eUndefined);
    extent = vk::Extent3D{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
    };

    std::vector<uint32_t> connected_binding_indices;

    uint32_t binding_index = 0;
    for (const auto& input : input_connectors) {
        if (io_layout.is_connected(input)) {
            const vk::ImageCreateInfo create_info = io_layout[input]->get_create_info_or_throw();
            if (format == vk::Format::eUndefined)
                format = create_info.format;
            extent = min(extent, create_info.extent);
            connected_binding_indices.emplace_back(binding_index);
        }
        spec_builder.add_entry(io_layout.is_connected(input));
        binding_index++;
    }

    if (connected_binding_indices.empty()) {
        throw graph_errors::node_error{"at least one input must be connected."};
    }

    spec_info = spec_builder.build();

    // -------------------------------------------------------

    source = R"(
#version 460
#extension GL_GOOGLE_include_directive    : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

)";

    for (const auto& binding_index : connected_binding_indices) {
        source.append(fmt::format("layout(set = 0, binding = {}) uniform sampler2D img_{:02};\n",
                                  binding_index, binding_index));
    }

    source.append(fmt::format(
        "layout(set = 0, binding = {}) uniform writeonly restrict image2D img_output;\n",
        binding_index));

    source.append(R"(
void main() {
    const ivec2 ipos = ivec2(gl_GlobalInvocationID);
    if (any(greaterThanEqual(ipos, imageSize(img_output)))) return;

    vec4 accumulator = )");
    source.append(fmt::format("{};\n", !initial_value.empty()
                                           ? initial_value
                                           : fmt::format("texelFetch(img_{:02}, ipos, 0)",
                                                         connected_binding_indices.front())));

    for (uint32_t i = initial_value.empty() ? 1 : 0; i < connected_binding_indices.size(); i++) {
        std::string result_expression = std::regex_replace(
            reduction, std::regex("current_value"),
            fmt::format("texelFetch(img_{:02}, ipos, 0)", connected_binding_indices[i]));
        result_expression =
            std::regex_replace(result_expression, std::regex("initial_value"), initial_value);
        source.append(fmt::format("    accumulator = {};\n", result_expression));
    }
    source.append(R"(

    imageStore(img_output, ipos, accumulator);
}
)");

    const auto& shader_compiler = ShaderCompiler::get(context);
    shader = shader_compiler->compile_glsl_to_shadermodule(context, source, "<memory>add.comp",
                                                           vk::ShaderStageFlagBits::eCompute);

    // -------------------------------------------------------

    return {
        ManagedVkImageOut::compute_write("out", format, extent),
    };
}

SpecializationInfoHandle
Reduce::get_specialization_info([[maybe_unused]] const NodeIO& io) noexcept {
    return spec_info;
}

// const void* ReduceInputsNode::get_push_constant([[maybe_unused]] GraphRun& run, [[maybe_unused]]
// const NodeIO& io) {
//     return &pc;
// }

std::tuple<uint32_t, uint32_t, uint32_t>
Reduce::get_group_count([[maybe_unused]] const merian_nodes::NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle Reduce::get_shader_module() {
    return shader;
}

Reduce::NodeStatusFlags Reduce::properties(Properties& props) {
    bool needs_reconnect = false;

    needs_reconnect |=
        props.config_text("initial value", initial_value, true,
                          "a GLSL expression that yields a vec4. Or empty to use the first input.");
    needs_reconnect |=
        props.config_text("reduction", reduction, true,
                          "a GLSL expression that yields a vec4. You can use the variables "
                          "initial_value, accumulator, and current_value.");

    needs_reconnect |= props.config_uint("number inputs", number_inputs, "");
    props.output_text("output extent: {}x{}x{}", extent.width, extent.height, extent.depth);

    if (props.st_begin_child("show_source", "show source")) {
        props.output_text(source);
        props.st_end_child();
    }

    if (needs_reconnect) {
        return NodeStatusFlagBits::NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian_nodes
