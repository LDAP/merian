#include "merian-nodes/nodes/shadertoy/shadertoy.hpp"

#include "merian-nodes/connectors/managed_vk_image_out.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#if MERIAN_ENABLE_SHADERC
#include "merian/vk/shader/shader_compiler_shaderc.hpp"
#endif

namespace merian_nodes {

static const char* shadertoy_pre = R"(#version 460
#extension GL_EXT_scalar_block_layout : require

// Use constants to set local size
layout(local_size_x_id = 0, local_size_y_id = 1) in;

layout(binding = 0, set = 0, rgba32f) uniform restrict image2D result;
layout(push_constant) uniform constants {
    vec2 iResolution;
    float iTime;
    float iTimeDelta;
    float iFrame;
};

)";

static const char* shadertoy_post = R"(vec4 _merian_shadertoy_toLinear(vec4 sRGB)
{
    bvec4 cutoff = lessThan(sRGB, vec4(0.04045));
    vec4 higher = pow((sRGB + vec4(0.055))/vec4(1.055), vec4(2.4));
    vec4 lower = sRGB/vec4(12.92);

    return mix(higher, lower, cutoff);
}

void main()
{
  const uvec2 pixel = gl_GlobalInvocationID.xy;
  if((pixel.x >= iResolution.x) || (pixel.y >= iResolution.y))
  {
    return;
  }

  vec4 frag_color;
  mainImage(frag_color, vec2(pixel.x, iResolution.y - pixel.y - 1));
  // WebGL or Shadertoy does not do a Linear->sRGB conversion
  // thus the shader must output sRGB. But here the shader is expected to output
  // linear!
  imageStore(result, ivec2(pixel), _merian_shadertoy_toLinear(frag_color));
}

)";

static const char* default_shader = R"(
void mainImage(out vec4 fragColor, in vec2 fragCoord) { 
    fragColor = vec4(vec3(0), 1.);
}
)";

class ShadertoyInjectCompiler : public ShaderCompiler {
  public:
    ShadertoyInjectCompiler(const ShaderCompilerHandle forwarding_compiler)
        : forwarding_compiler(forwarding_compiler) {}

    ~ShadertoyInjectCompiler() {}

    std::vector<uint32_t> compile_glsl(const std::string& source,
                                       const std::string& source_name,
                                       const vk::ShaderStageFlagBits shader_kind) final {
        SPDLOG_INFO("(re-)compiling {}", source_name);
        return forwarding_compiler->compile_glsl(shadertoy_pre + source + shadertoy_post,
                                                 source_name, shader_kind);
    }

  private:
    const ShaderCompilerHandle forwarding_compiler;
};

Shadertoy::Shadertoy(const ContextHandle context)
    : AbstractCompute(context, sizeof(PushConstant)), shader_glsl(default_shader) {

    ShaderCompilerHandle shaderc_compiler = nullptr;
#if MERIAN_ENABLE_SHADERC
    shaderc_compiler = std::make_shared<merian::ShadercCompiler>();
#endif

    if (!shaderc_compiler) {
        return;
    }

    compiler = std::make_shared<ShadertoyInjectCompiler>(shaderc_compiler);
    reloader = std::make_unique<HotReloader>(context, compiler);
    shader = compiler->compile_glsl_to_shadermodule(context, shader_glsl, "<memory>Shadertoy.comp",
                                                    vk::ShaderStageFlagBits::eCompute);

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();
}

std::vector<OutputConnectorHandle>
Shadertoy::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    if (!reloader) {
        throw graph_errors::node_error{"the shaderc feature must be enabled for this node."};
    }

    if (shader_source_selector == 0) {
        if (error) {
            throw graph_errors::node_error{error->what()};
        }
    } else if (shader_source_selector == 1) {
        if (resolved_shader_path.empty()) {
            throw graph_errors::node_error{"no shader path is set."};
        }
        if (!std::filesystem::exists(resolved_shader_path)) {
            throw graph_errors::node_error{
                fmt::format("file does not exist: {}", resolved_shader_path.string())};
        }
    }

    constant.iResolution = {extent.width, extent.height};
    return {ManagedVkImageOut::compute_write("out", vk::Format::eR8G8B8A8Unorm, extent)};
}

SpecializationInfoHandle
Shadertoy::get_specialization_info([[maybe_unused]] const NodeIO& io) noexcept {
    return spec_info;
}

const void* Shadertoy::get_push_constant([[maybe_unused]] GraphRun& run,
                                         [[maybe_unused]] const NodeIO& io) {
    constant.iTimeDelta = static_cast<float>(run.get_time_delta());
    constant.iTime = static_cast<float>(run.get_elapsed());
    constant.iFrame++;

    return &constant;
}

std::tuple<uint32_t, uint32_t, uint32_t>
Shadertoy::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle Shadertoy::get_shader_module() {
    if (shader_source_selector == 1) {
        try {
            shader = reloader->get_shader(resolved_shader_path, vk::ShaderStageFlagBits::eCompute);
            error.reset();
        } catch (const ShaderCompiler::compilation_failed& e) {
            error = e;
        }
    }

    return shader;
}

AbstractCompute::NodeStatusFlags Shadertoy::properties(Properties& config) {
    bool needs_reconnect = false;
    bool needs_compile = false;

    if (config.config_options("shader source", shader_source_selector, {"inline", "file"},
                              Properties::OptionsStyle::COMBO)) {
        needs_reconnect = true;
        if (shader_source_selector == 0) {
            needs_compile = true;
        }
        error.reset();
    }

    switch (shader_source_selector) {
    case 0: {
        if (config.config_text_multiline("shader", shader_glsl, false)) {
            needs_compile = true;
        }
        if (reloader) {
            reloader->clear();
        }
        break;
    }
    case 1: {
        if (config.config_text("shader path", shader_path, true)) {
            needs_reconnect = true;
            resolved_shader_path =
                context->file_loader.find_file(shader_path).value_or(shader_path);
        }

        if (std::filesystem::exists(resolved_shader_path)) {
            if (config.config_bool("convert to inline")) {
                shader_source_selector = 0;
                shader_glsl = FileLoader::load_file(resolved_shader_path);
                needs_compile = true;
            }
        }
        break;
    }
    default:
        assert(0);
    }

    if (error) {
        config.st_separate("Compilation failed.");
        config.output_text(error->what());
    }

    if (compiler && needs_compile) {
        try {
            shader = compiler->compile_glsl_to_shadermodule(
                context, shader_glsl, "<memory>Shadertoy.comp", vk::ShaderStageFlagBits::eCompute);
            error.reset();
        } catch (const ShaderCompiler::compilation_failed& e) {
            error = e;
        }
    }

    config.st_separate();

    needs_reconnect |= config.config_uint("width", extent.width, "");
    needs_reconnect |= config.config_uint("height", extent.height, "");

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    } else {
        return {};
    }
}

} // namespace merian_nodes
