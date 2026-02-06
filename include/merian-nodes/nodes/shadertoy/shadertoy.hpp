#pragma once

#include "merian-nodes/nodes/compute_node/compute_node.hpp"

#include "merian/shader/shader_compiler.hpp"
#include "merian/shader/shader_hotreloader.hpp"

namespace merian {

// A generator node that pushes the Shadertoy variables as push constant.
class Shadertoy : public AbstractCompute {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  private:
    struct PushConstant {
        merian::float2 iResolution{};
        float iTime{};
        float iTimeDelta{};
        int32_t iFrame{};
        merian::float4 iMouse{};
        merian::float4 iDate{};
    };

  public:
    Shadertoy();

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    // none if shader compiler is not available.
    GLSLShaderCompilerHandle compiler = nullptr;
    // none if shader compiler is not available.
    std::unique_ptr<HotReloader> reloader = nullptr;

    int shader_source_selector = 0;
    std::string shader_glsl;
    std::string shader_path = {0};
    std::filesystem::path resolved_shader_path;

    vk::Extent3D extent = {1920, 1080, 1};

    SpecializationInfoHandle spec_info;
    VulkanEntryPointHandle shader;
    std::optional<GLSLShaderCompiler::compilation_failed> error;

    PushConstant constant;

    ShaderCompileContextHandle compile_context;
};

} // namespace merian
