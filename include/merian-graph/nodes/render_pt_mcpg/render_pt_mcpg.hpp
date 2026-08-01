#pragma once

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/render_pt_mcpg/mcpg.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/gbuffer.hpp"
#include "merian-shaders/light-cache/hashed_irradiance_cache.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing.hpp"
#include "merian/vk/raytrace/shader_binding_table.hpp"

#include <array>

namespace merian {

// Markov-chain path-guiding renderer. Currently a plain BSDF-sampled path tracer; the guiding is
// WIP.
class RenderMCPG : public Node {

  public:
    RenderMCPG();

    ~RenderMCPG() override = default;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 Submission& submission) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    void ensure_pipeline(const SceneHandle& scene);
    void update_render_constants();

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    // Connectors
    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();
    ManagedVkImageOutHandle con_irradiance;
    ManagedVkImageOutHandle con_debug;

    // Owns its own persistent buffer + shader binding (composed into this node's program).
    HashedIrradianceCacheHandle irr_cache;
    MCPGHandle mcpg;

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};

    // --- Surface transport (link-time constants) ---
    int32_t spp = 1;
    int32_t max_path_length = 8;
    int32_t emitted_max_path_length = max_path_length;
    bool emission_on_primary = true;
    bool demodulate_albedo = false;
    bool reference_mode = false;
    std::array<bool, 8> mask_enabled{true, true, true, true, true, true, true, true};

    // --- Guiding Markov chain ---
    float dir_guide_prior = 0.2f;
    int32_t mc_samples = 5;
    float p_guiding = 0.85f; // probability to sample the guiding distribution instead of the BSDF
    bool missing_light_heuristic = true;

    uint32_t mc_adaptive_buffer_size = 32777259;
    bool mc_split_storage = true;

    // --- Light cache ---
    bool use_light_cache_tail = false;
    uint32_t lc_buffer_size = 4000037;
    uint32_t lc_probe_count = 4;
    bool lc_stochastic_interpolation = false;
    bool lc_split_storage = true;
    float lc_min_pdf = 1.0f;

    // --- Misc ---
    int32_t debug_output_selector = 0;

    // Slang program + pipeline; rebuilt when the scene composition changes.
    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> entry_point;
    Versioned<Pipeline> pipeline;
    Versioned<ShaderBindingTable> sbt; // only used when use_raygen
    Versioned<ShaderObject> params;

    // Executor: compute (faster, no RT-pipeline VGPR cap) or raygen (enables SER-only experiments).
    bool use_raygen = false;
};

} // namespace merian
