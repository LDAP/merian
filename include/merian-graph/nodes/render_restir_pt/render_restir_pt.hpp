#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/gbuffer.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/raytrace/shader_binding_table.hpp"

#include <array>

namespace merian {

// ReSTIR path tracing after Lin et al. (2022) with the algorithms of ReSTIR PT Enhanced
// (Lin et al. 2026): the hybrid shift with footprint-based reconnection, reciprocal neighbor
// pairs, and a history cap that follows sample duplication. Compatibility-guided neighbor
// selection (Junkins et al. 2026) replaces the pairs where selected.
class RenderRestirPT : public Node {

  public:
    enum Pass {
        Initial = 0,
        TemporalNeighbors = 1,
        SpatialNeighbors = 2,
        Shift = 3,
        Temporal = 4,
        Spatial = 5,
        Duplication = 6,
        Splat = 7,
        SubpixelBackprojection = 8,
        CoverageNeighbors = 9,
        PassCount = 10
    };

    RenderRestirPT();

    ~RenderRestirPT() override = default;

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
    void upload_pairing(Submission& submission);

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    GBufferInHandle con_gbuffer;
    GBufferInHandle con_prev_gbuffer;
    VkBufferInHandle con_prev_reservoirs = VkBufferIn::create();
    ManagedVkImageOutHandle con_irradiance;
    ManagedVkBufferOutHandle con_reservoirs;

    vk::Format irradiance_format = vk::Format::eR32G32B32A32Sfloat;
    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};

    // initial candidates
    int32_t spp = 1;
    int32_t max_path_length = 6;
    int32_t emitted_max_path_length = max_path_length;
    uint32_t seed = 0;
    bool russian_roulette = true;
    bool emission_on_primary = true;
    bool area = false;
    bool demodulate_albedo = false;
    bool use_raygen = true;
    std::array<bool, 8> mask_enabled{true, true, true, true, true, true, true, true};

    // shift
    int32_t shift_mapping = 1;
    // c of the footprint criterion, in percent of the primary footprint
    float min_footprint = 0.02f;
    float footprint_jitter = 0.2f;
    float min_roughness = 0.2f;
    float roughness_jitter = 0.f;

    // temporal reuse
    bool temporal_enable = true;
    float history_cap = 20.f;
    bool stochastic_backprojection = false;
    bool disocclusion_motion = false;
    bool specular_motion = false;
    float specular_motion_roughness = 0.1f;
    bool dynamic_scene = false;
    int32_t temporal_mode = 0;
    bool duplication_cap = false;
    float duplication_exponent = 0.1f;

    // spatial reuse
    int32_t spatial_rounds = 1;
    int32_t neighbor_selection = 0;
    int32_t neighbor_count = 3;
    float spatial_radius = 30.f;
    bool geometry_rejection = true;
    float reject_normal = 0.5f;
    float reject_depth = 0.1f;
    bool decoupled_shading = false;

    // compatibility-guided neighbor selection
    int32_t cgns_candidates = 32;
    float scale_solid_angle = 0.05f;
    float normal_beta = 8.f;
    bool early_stopping = true;
    float early_stop_cutoff = 0.5f;

    int32_t debug_view = 0;

    SlangCompositionHandle gbuffer_composition;
    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    std::array<Versioned<SlangProgramEntryPoint>, PassCount> entry_points;
    std::array<Versioned<Pipeline>, PassCount> pipelines;
    Versioned<ShaderBindingTable> initial_sbt;
    std::array<Versioned<ShaderObject>, PassCount> params;

    BufferHandle pong_reservoirs;
    BufferHandle history;
    BufferHandle splat_counts;
    BufferHandle shifts;
    BufferHandle neighbors;
    BufferHandle confidence_sums;
    BufferHandle backprojected;
    BufferHandle temporal_domains;
    BufferHandle splat_samples;
    BufferHandle queue;
    BufferHandle pairing;
    std::array<BufferHandle, 2> duplication;
    // the pairing tables are regenerated when their parameters change
    bool pairing_dirty = true;
};

} // namespace merian
