#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-shaders/gbuffer.hpp"
#include "merian-shaders/light-cache/hashed_irradiance_cache.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
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
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    void update_render_constants();

    vk::BufferCreateInfo mc_state_buffer_create_info() const;

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    // Connectors
    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    PtrInHandle<GBuffer> con_gbuffer = PtrIn<GBuffer>::create();
    ManagedVkImageOutHandle con_irradiance;
    ManagedVkImageOutHandle con_debug;
    // Persistent Markov-chain state, bound globally and updated in place across frames.
    ManagedVkBufferOutHandle con_mc_states;

    // Owns its own persistent buffer + shader binding (composed into this node's program).
    HashedIrradianceCacheHandle irr_cache;

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};

    // --- Surface transport (link-time constants) ---
    int32_t spp = 1;
    int32_t max_path_length = 8;
    int32_t emitted_max_path_length = max_path_length;
    bool emission_on_primary = false;
    bool reference_mode = false;
    std::array<bool, 8> mask_enabled{true, true, true, true, true, true, true, true};

    // --- Guiding Markov chain ---
    float dir_guide_prior = 0.2f;
    int32_t mc_samples = 5;
    float mc_samples_adaptive_prob = 0.7f;
    float p_guiding = 0.85f; // probability to sample the guiding distribution instead of the BSDF
    bool missing_light_heuristic = true;

    int32_t mc_adaptive_grid_type = 0;
    uint32_t mc_adaptive_buffer_size = 32777259;
    float mc_adaptive_grid_tan_alpha_half = 0.003f;
    float mc_adaptive_grid_steps_per_unit_size = 6.0f;
    float mc_adaptive_grid_min_width = 0.01f;
    float mc_adaptive_grid_power = 4.0f;

    uint32_t mc_static_buffer_size = 800009;
    float mc_static_grid_width = 25.3f;

    // --- Light cache ---
    bool use_light_cache_tail = false;

    // --- Misc ---
    uint32_t seed = 0;
    bool randomize_seed = true;
    int32_t debug_output_selector = 0;

    // Slang program + pipeline; rebuilt when the scene composition changes.
    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> entry_point;
    Versioned<RayTracingPipeline> pipeline;
    Versioned<ShaderBindingTable> sbt;
    Versioned<ShaderObject> params;
    Versioned<ShaderObject> globals;
    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;
};

} // namespace merian
