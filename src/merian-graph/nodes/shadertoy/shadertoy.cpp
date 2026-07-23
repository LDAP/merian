#include "merian-graph/nodes/shadertoy/shadertoy.hpp"

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/errors.hpp"
#include "merian/io/file_loader.hpp"

namespace merian {

// The user body is GLSL-flavored (mainImage(out vec4, in vec2)); slang's glsl module keeps the
// common Shadertoy vocabulary (vec*, mix, fract, ...) valid.
static const char* shadertoy_pre = R"(import glsl;

RWTexture2D<float4> out_out;

[vk::push_constant]
cbuffer constants {
    float4 iMouse;
    float4 iDate;
    float2 iResolution;
    float iTime;
    float iTimeDelta;
    int iFrame;
};

)";

// numthreads is injected between these two halves so it tracks local_size_x/y.
static const char* shadertoy_post_head = R"(
float4 _merian_shadertoy_toLinear(float4 sRGB)
{
    const float4 higher = pow((sRGB + 0.055) / 1.055, 2.4);
    const float4 lower = sRGB / 12.92;
    return select(sRGB < 0.04045, lower, higher);
}

)";

static const char* shadertoy_main = R"([shader("compute")]
void main(uint3 _merian_tid: SV_DispatchThreadID)
{
    const uint2 pixel = _merian_tid.xy;
    if ((pixel.x >= uint(iResolution.x)) || (pixel.y >= uint(iResolution.y)))
        return;

    vec4 frag_color;
    mainImage(frag_color, vec2(pixel.x, iResolution.y - pixel.y - 1));
    // WebGL or Shadertoy does not do a Linear->sRGB conversion
    // thus the shader must output sRGB. But here the shader is expected to output
    // linear!
    out_out.Store(int2(pixel), _merian_shadertoy_toLinear(frag_color));
}
)";

static const char* default_shader = R"(
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(vec3(0), 1.);
}
)";

Shadertoy::Shadertoy() : AbstractCompute(sizeof(PushConstant)), shader_glsl(default_shader) {}

DeviceSupportInfo Shadertoy::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(query_info, {}, {}, {}, {}, {"Shader", "ImageQuery"}, {}, {},
                                    {});
}

std::vector<InputConnectorDescriptor> Shadertoy::describe_inputs() {
    return {{"controller", con_controller, {}, 0, true}};
}

std::string Shadertoy::current_body() const {
    if (shader_source_selector == 1) {
        return FileLoader::load_file_as_string(resolved_shader_path);
    }
    return shader_glsl;
}

std::string Shadertoy::compose_source(const std::string& body) {
    return shadertoy_pre + body + shadertoy_post_head +
           fmt::format("[numthreads({}, {}, 1)]\n", local_size_x, local_size_y) + shadertoy_main;
}

bool Shadertoy::try_compile(const std::string& body) {
    try {
        const auto composition = SlangComposition::create();
        composition->add_module_from_string("merian_shadertoy", compose_source(body), true);
        SlangProgram::create(compile_context, composition).get();
        error.reset();
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

std::vector<OutputConnectorDescriptor>
Shadertoy::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    if (shader_source_selector == 1) {
        if (resolved_shader_path.empty()) {
            throw graph_errors::node_error{"no shader path is set."};
        }
        if (!std::filesystem::exists(resolved_shader_path)) {
            throw graph_errors::node_error{
                fmt::format("file does not exist: {}", resolved_shader_path.string())};
        }
        last_write_time = std::filesystem::last_write_time(resolved_shader_path);
    }

    if (!try_compile(current_body())) {
        throw graph_errors::node_error{*error};
    }

    constant.iResolution = {extent.width, extent.height};
    return {{"out", ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, extent),
             ConnectorAccess::compute_write}};
}

SlangCompositionHandle Shadertoy::create_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_string("merian_shadertoy", compose_source(current_body()), true);
    return composition;
}

void Shadertoy::write_constants([[maybe_unused]] GraphRun& run,
                                [[maybe_unused]] const NodeIO& io,
                                [[maybe_unused]] ShaderCursor& cursor) {
    // hot reload: pick up edits to the shader file; keep the last working shader on error
    if (shader_source_selector != 1 || !std::filesystem::exists(resolved_shader_path)) {
        return;
    }
    const auto write_time = std::filesystem::last_write_time(resolved_shader_path);
    if (write_time == last_write_time) {
        return;
    }
    last_write_time = write_time;
    if (try_compile(current_body())) {
        invalidate_shader();
    }
}

const void* Shadertoy::get_push_constant([[maybe_unused]] GraphRun& run,
                                         [[maybe_unused]] const NodeIO& io) {
    // iMouse from the optional controller (xy: current, zw: click; y measured from the bottom).
    if (io.is_connected(con_controller)) {
        const InputControllerHandle& controller = io[con_controller];
        if (controller && controller != registered_controller.lock()) {
            controller->add_listener(mouse_input);
            registered_controller = controller;
        }
        const float h = static_cast<float>(extent.height);
        constant.iMouse.x = mouse_input->x;
        constant.iMouse.y = h - mouse_input->y;
        constant.iMouse.z = mouse_input->down ? mouse_input->click_x : -mouse_input->click_x;
        constant.iMouse.w = (h - mouse_input->click_y) * (mouse_input->down ? 1.0f : -1.0f);
    }

    constant.iTimeDelta = static_cast<float>(run.get_time_delta());
    constant.iTime = static_cast<float>(run.get_elapsed());
    constant.iFrame = static_cast<int32_t>(run.get_total_iteration());

    const auto now = std::chrono::system_clock::now();
    const auto now_d = std::chrono::floor<std::chrono::days>(now);
    const std::chrono::year_month_day ymd(now_d);
    constant.iDate.x = static_cast<float>(static_cast<int>(ymd.year()));
    constant.iDate.y = static_cast<float>(static_cast<unsigned int>(ymd.month()));
    constant.iDate.z = static_cast<float>(static_cast<unsigned int>(ymd.day()));
    constant.iDate.w =
        std::chrono::duration_cast<std::chrono::duration<float>>(now - now_d).count();

    return &constant;
}

std::tuple<uint32_t, uint32_t, uint32_t>
Shadertoy::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

AbstractCompute::NodeStatusFlags Shadertoy::properties(Properties& config) {
    bool needs_reconnect = false;

    if (config.config_options("shader source", shader_source_selector, {"inline", "file"},
                              Properties::OptionsStyle::COMBO)) {
        needs_reconnect = true;
        error.reset();
    }

    switch (shader_source_selector) {
    case 0: {
        if (config.config_text_multiline("shader", shader_glsl, false)) {
            if (try_compile(shader_glsl)) {
                invalidate_shader();
            }
        }
        break;
    }
    case 1: {
        if (config.config_text("shader path", shader_path, true)) {
            needs_reconnect = true;
            resolved_shader_path =
                context->get_file_loader()->find_file(shader_path).value_or(shader_path);
        }

        if (std::filesystem::exists(resolved_shader_path)) {
            if (config.config_bool("convert to inline")) {
                shader_source_selector = 0;
                shader_glsl = FileLoader::load_file_as_string(resolved_shader_path);
                if (try_compile(shader_glsl)) {
                    invalidate_shader();
                }
            }
        }
        break;
    }
    default:
        assert(0);
    }

    if (error) {
        config.st_separate("Compilation failed.");
        config.output_text(*error);
    }

    config.st_separate();

    needs_reconnect |= config.config_uint("width", extent.width, "");
    needs_reconnect |= config.config_uint("height", extent.height, "");

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
