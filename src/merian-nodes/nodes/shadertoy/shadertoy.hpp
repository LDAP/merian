#pragma once

#include "glm/ext/vector_float2.hpp"

#include "merian-nodes/nodes/compute_node/compute_node.hpp"

#include "merian/utils/stopwatch.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_hotreloader.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian_nodes {

// A generator node that pushes the Shadertoy variables as push constant.
class Shadertoy : public AbstractCompute {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  private:
    struct PushConstant {
        glm::vec2 iResolution{};
        float iTime{};
        float iTimeDelta{};
        float iFrame{};
    };

  public:
    Shadertoy(const ContextHandle context);

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    SpecializationInfoHandle get_specialization_info(const NodeIO& io) noexcept override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    // none if shader compiler is not available.
    ShaderCompilerHandle compiler = nullptr;
    // none if shader compiler is not available.
    std::unique_ptr<HotReloader> reloader = nullptr;

    int shader_source_selector = 0;
    std::string shader_glsl;
    std::string shader_path = {0};
    std::filesystem::path resolved_shader_path;

    vk::Extent3D extent = {1920, 1080, 1};

    ShaderModuleHandle shader;
    SpecializationInfoHandle spec_info;
    std::optional<ShaderCompiler::compilation_failed> error;

    PushConstant constant;
    Stopwatch sw;
};

} // namespace merian_nodes
