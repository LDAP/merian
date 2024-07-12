#include "shadertoy.hpp"

#include "merian-nodes/connectors/managed_vk_image_out.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

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
  mainImage(frag_color, pixel);
  // WebGL or Shadertoy does not do a Linear->sRGB conversion
  // thus the shader must output sRGB. But here the shader is expected to output
  // linear!
  imageStore(result, ivec2(pixel), _merian_shadertoy_toLinear(frag_color));
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

Shadertoy::Shadertoy(const SharedContext context,
                     const std::string& path,
                     const ShaderCompilerHandle& compiler)
    : AbstractCompute(context, sizeof(PushConstant)), shader_path(path),
      reloader(context, std::make_shared<ShadertoyInjectCompiler>(compiler)) {

    sw.reset();
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();
}

std::vector<OutputConnectorHandle>
Shadertoy::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
    constant.iResolution = {extent.width, extent.height};
    return {ManagedVkImageOut::compute_write("out", vk::Format::eR8G8B8A8Unorm, extent)};
}

AbstractCompute::NodeStatusFlags Shadertoy::pre_process([[maybe_unused]] GraphRun& run,
                                                        [[maybe_unused]] const NodeIO& io) {
    NodeStatusFlags flags{};
    if (requires_rebuild) {
        flags |= NodeStatusFlagBits::NEEDS_RECONNECT;
    }
    requires_rebuild = false;
    return flags;
}

SpecializationInfoHandle
Shadertoy::get_specialization_info([[maybe_unused]] const NodeIO& io) const noexcept {
    return spec_info;
}

const void* Shadertoy::get_push_constant([[maybe_unused]] GraphRun& run,
                                         [[maybe_unused]] const NodeIO& io) {
    float new_time = sw.seconds();
    constant.iTimeDelta = new_time - constant.iTime;
    constant.iTime = new_time;
    constant.iFrame++;

    return &constant;
}

std::tuple<uint32_t, uint32_t, uint32_t>
Shadertoy::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle Shadertoy::get_shader_module() {
    ShaderModuleHandle shader;

    try {
        shader = reloader.get_shader(shader_path, vk::ShaderStageFlagBits::eCompute);
        error.reset();
    } catch (const ShaderCompiler::compilation_failed& e) {
        error = e;
    }

    return shader;
}

AbstractCompute::NodeStatusFlags Shadertoy::properties(Properties& config) {
    vk::Extent3D old_extent = extent;
    config.config_uint("width", extent.width, "");
    config.config_uint("height", extent.height, "");

    if (error) {
        config.st_separate("Compilation failed.");
        config.output_text(error->what());
    }

    if (old_extent != extent) {
        return NEEDS_RECONNECT;
    } else {
        return {};
    }
}

} // namespace merian_nodes
