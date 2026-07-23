#include "merian-graph/nodes/reduce/reduce.hpp"

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian/vk/utils/math.hpp"

#include <atomic>
#include <regex>

namespace merian {

namespace {
std::string next_module_name() {
    static std::atomic<uint64_t> counter = 0;
    return fmt::format("merian_reduce_generated_{}", counter++);
}
} // namespace

Reduce::Reduce() : AbstractCompute(), module_name(next_module_name()) {}

Reduce::~Reduce() {}

DeviceSupportInfo Reduce::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(query_info, {}, {}, {}, {}, {"Shader", "ImageQuery"}, {}, {},
                                    {});
}

std::vector<InputConnectorDescriptor> Reduce::describe_inputs() {
    if (input_connectors.size() != number_inputs) {
        input_connectors.clear();
        for (uint32_t i = 0; i < number_inputs; i++) {
            input_connectors.emplace_back(VkSampledImageIn::create());
        }
    }

    std::vector<InputConnectorDescriptor> descriptors;
    for (uint32_t i = 0; i < number_inputs; i++) {
        descriptors.push_back(
            {fmt::format("input_{}", i), input_connectors[i], ConnectorAccess::compute_read, 0, true});
    }
    return descriptors;
}

std::vector<OutputConnectorDescriptor> Reduce::describe_outputs(const NodeIOLayout& io_layout) {
    vk::Format format = output_format.value_or(vk::Format::eUndefined);
    extent = vk::Extent3D{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
    };

    connected_indices.clear();

    uint32_t input_index = 0;
    for (const auto& input : input_connectors) {
        if (io_layout.is_connected(input)) {
            const vk::ImageCreateInfo create_info = io_layout[input]->get_create_info_or_throw();
            if (format == vk::Format::eUndefined)
                format = create_info.format;
            extent = min(extent, create_info.extent);
            connected_indices.emplace_back(input_index);
        }
        input_index++;
    }

    if (connected_indices.empty()) {
        throw graph_errors::node_error{"at least one input must be connected."};
    }

    // Retire the shared session before validating so try_compile compiles the new source fresh and
    // caches it under this instance's module name; create_composition then reuses that compile.
    std::string new_source = generate_source();
    if (new_source != source) {
        source = std::move(new_source);
        invalidate_shader();
    }
    if (!try_compile(source)) {
        throw graph_errors::node_error{*error};
    }

    return {
        {"out", ManagedVkImageOut::create(format, extent), ConnectorAccess::compute_write},
    };
}

std::string Reduce::generate_source() const {
    // GLSL-style vector constructors keep user expressions from older configs valid
    std::string new_source = R"(typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;

)";

    for (const auto& input_index : connected_indices) {
        new_source.append(fmt::format("Sampler2D<float4> in_input_{};\n", input_index));
    }
    new_source.append("RWTexture2D<float4> out_out;\n");

    new_source.append(fmt::format(R"(
[numthreads({}, {}, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {{
    uint out_w, out_h;
    out_out.GetDimensions(out_w, out_h);
    if (ipos.x >= int(out_w) || ipos.y >= int(out_h))
        return;

    vec4 accumulator = )",
                                  local_size_x, local_size_y));
    new_source.append(
        fmt::format("{};\n", !initial_value.empty()
                                 ? initial_value
                                 : fmt::format("in_input_{}[ipos]", connected_indices.front())));

    const std::regex current_value_re("current_value");
    const std::regex initial_value_re("initial_value");
    for (uint32_t i = initial_value.empty() ? 1 : 0; i < connected_indices.size(); i++) {
        std::string result_expression = std::regex_replace(
            reduction, current_value_re, fmt::format("in_input_{}[ipos]", connected_indices[i]));
        result_expression = std::regex_replace(result_expression, initial_value_re, initial_value);
        new_source.append(fmt::format("    accumulator = {};\n", result_expression));
    }
    new_source.append(R"(
    out_out.Store(ipos, accumulator);
}
)");
    return new_source;
}

bool Reduce::try_compile(const std::string& source_candidate) {
    try {
        const auto composition = SlangComposition::create();
        composition->add_module_from_string(module_name, source_candidate, true);
        SlangProgram::create(compile_context, composition).get();
        error.reset();
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

SlangCompositionHandle Reduce::create_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_string(module_name, source, true);
    return composition;
}

std::tuple<uint32_t, uint32_t, uint32_t>
Reduce::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

Reduce::NodeStatusFlags Reduce::properties(Properties& props) {
    bool needs_reconnect = false;

    bool expressions_changed = false;
    expressions_changed |=
        props.config_text("initial value", initial_value, true,
                          "an expression that yields a vec4. Or empty to use the first input.");
    expressions_changed |=
        props.config_text("reduction", reduction, true,
                          "an expression that yields a vec4. You can use the variables "
                          "initial_value, accumulator, and current_value.");
    if (expressions_changed) {
        // keep the last working source running on error
        needs_reconnect |= connected_indices.empty() || try_compile(generate_source());
    }

    if (error) {
        props.st_separate("Compilation failed.");
        props.output_text(*error);
    }

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

} // namespace merian
