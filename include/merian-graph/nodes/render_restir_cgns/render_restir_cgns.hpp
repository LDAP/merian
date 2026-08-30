#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/gbuffer.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/raytrace/shader_binding_table.hpp"

#include <array>

namespace merian {

// ReSTIR path tracing with compatibility-guided neighbor selection, after Junkins et al. (2026).
// Paths reconnect at their second vertex; every domain is a GBuffer pixel, so a shift reuses the
// destination's primary hit instead of retracing a primary ray.
class RenderRestirCGNS : public Node {

  public:
    enum Pass { Initial = 0, Temporal = 1, SelectNeighbors = 2, Spatial = 3, PassCount = 4 };

    // How the previous frame's reservoirs reach a pixel.
    enum class TemporalMode { Gather = 0, Splat = 1, SplatGather = 2 };

    RenderRestirCGNS();

    ~RenderRestirCGNS() override = default;

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
    [[nodiscard]] vk::BufferCreateInfo reservoir_buffer_create_info() const;
    [[nodiscard]] vk::BufferCreateInfo reconnection_buffer_create_info() const;
    [[nodiscard]] vk::BufferCreateInfo neighbor_buffer_create_info() const;

    [[nodiscard]] vk::BufferCreateInfo splat_buffer_create_info() const;

    [[nodiscard]] vk::BufferCreateInfo splat_count_buffer_create_info() const;

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();
    ShaderObjectInHandle<GBufferObject> con_prev_gbuffer = ShaderObjectIn<GBufferObject>::create();
    VkBufferInHandle con_prev_reservoirs = VkBufferIn::create();
    VkBufferInHandle con_prev_reconnection = VkBufferIn::create();
    ManagedVkImageOutHandle con_irradiance;
    ManagedVkBufferOutHandle con_reservoirs;
    ManagedVkBufferOutHandle con_reconnection;

    vk::Format irradiance_format = vk::Format::eR32G32B32A32Sfloat;
    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};

    int32_t spp = 1;
    int32_t max_path_length = 5;
    int32_t emitted_max_path_length = max_path_length;
    uint32_t seed = 0;
    bool emission_on_primary = true;
    bool russian_roulette = true;
    bool use_raygen = true;
    bool demodulate_albedo = false;
    std::array<bool, 8> mask_enabled{true, true, true, true, true, true, true, true};

    bool temporal_enable = true;
    bool confidence_temporal = true;
    // [Liu et al. 2025]
    float confidence_cap = 20.f;

    int32_t spatial_rounds = 1;
    bool confidence_spatial = true;
    float spatial_radius = 30.f;
    bool geometry_rejection = true;
    float reject_normal = 0.5f;
    float reject_depth = 0.1f;

    // CGNS neighbor selection
    int32_t neighbor_count = 1;
    TemporalMode temporal_mode = TemporalMode::Gather;
    int32_t candidates = 32;
    float scale_solid_angle = 0.05f;
    float normal_beta = 8.f;
    bool early_stopping = true;
    float early_stop_cutoff = 0.5f;

    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    std::array<Versioned<SlangProgramEntryPoint>, PassCount> entry_points;
    std::array<Versioned<Pipeline>, PassCount> pipelines;
    Versioned<ShaderBindingTable> initial_sbt;
    std::array<Versioned<ShaderObject>, PassCount> params;

    BufferHandle pong_reservoirs;
    BufferHandle pong_reconnection;
    BufferHandle neighbors;
    BufferHandle splats;
    BufferHandle splat_counts;
};

} // namespace merian
