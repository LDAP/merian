#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-graph/objects/guiding_object.hpp"
#include "merian-graph/objects/path_record_sink.hpp"
#include "merian-shaders/gbuffer.hpp"
#include "merian-shaders/sampling/guiding.hpp"
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
#include <optional>

namespace merian {

class RenderPT : public Node {

  public:
    // match the enums in render_pt_common.slang
    enum class ScatterMode : int32_t { Off, MIS, RIS };
    enum class NEEMode : int32_t { Off, Mixture, Resampled };
    enum class DirectTarget : int32_t { Full, MIS, None };
    enum class DebugOutput : int32_t { ScatterStatistics, Guiding, NEE, DistanceGuiding, None };
    enum class TraceShader : int32_t { Auto, RayGeneration, Compute };

    RenderPT();

    ~RenderPT() override = default;

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
    struct Sampling {
        NEEMode nee_mode = NEEMode::Resampled;
        float shading_share = 0.45f;
        float nee_share = 0.1f;
        float guiding_share = 0.45f;
        bool cache_tail = false;

        float nee_probability(bool guided) const;
        float guided_probability(bool guided) const;
        bool properties(Properties& config,
                        const std::string& shading_label,
                        bool guided,
                        bool has_cache);
    };

    void ensure_pipeline(const SceneHandle& scene);
    void update_render_constants();
    uint32_t recorded_vertices_per_path() const;
    bool use_raygen() const;
    void update_guiding_slot();
    bool has_guiding() const;
    bool has_volume_guiding() const;
    float distance_guided_probability() const;
    bool has_irradiance_cache() const;

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    GBufferInHandle con_gbuffer;
    ShaderObjectInHandle<GuidingObject> con_guiding = ShaderObjectIn<GuidingObject>::create();
    ShaderObjectInHandle<DistanceGuidingObject> con_distance_guiding =
        ShaderObjectIn<DistanceGuidingObject>::create();
    ShaderObjectInHandle<IrradianceCacheObject> con_irradiance_cache =
        ShaderObjectIn<IrradianceCacheObject>::create();
    ManagedVkImageOutHandle con_irradiance;
    ManagedVkImageOutHandle con_debug;
    PathRecordSink path_records;
    ManagedVkImageOutHandle con_volume;
    ManagedVkImageOutHandle con_volume_depth;
    ManagedVkImageOutHandle con_volume_mv;
    VkSampledImageInHandle con_prev_volume_depth = VkSampledImageIn::create();

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};
    vk::Format irradiance_format = vk::Format::eR32G32B32A32Sfloat;
    int32_t spp = 1;
    uint32_t seed = 0;
    int32_t max_path_length = 5;
    int32_t max_diffuse_bounces = 16;
    int32_t max_glossy_bounces = 16;
    int32_t max_transmission_bounces = 16;
    int32_t emitted_max_path_length = max_path_length;
    bool emission_on_primary = true;
    bool enable_ser = false;
    TraceShader trace_shader = TraceShader::Auto;
    bool raygen_preferred = true;
    bool russian_roulette = true;
    bool demodulate_albedo = false;
    GuidingModelHandle guiding;
    uint32_t guiding_version = 0;
    DistanceGuidingModelHandle distance_guiding;
    uint32_t distance_guiding_version = 0;
    IrradianceCacheHandle irradiance_cache;
    uint32_t irradiance_cache_version = 0;

    bool volume_available = false;
    int32_t volume_spp = 1;
    bool volume_forward_project = true;
    float volume_forward_project_min_z = 50.f;
    vk::Format volume_depth_format = vk::Format::eR32Sfloat;

    DebugOutput debug_output = DebugOutput::ScatterStatistics;
    bool debug_connected = false;
    bool follow_specular = true;
    float follow_max_alpha = 0.0625f;
    ScatterMode scatter_mode = ScatterMode::MIS;
    int32_t scatter_candidates = 2;

    Sampling surface;
    int32_t surface_nee_bounces = 0;
    Sampling volume;
    float distance_transmittance_share = 0.1f;
    float distance_guiding_share = 0.9f;

    bool guiding_scale_with_alpha = true;
    float guiding_alpha_threshold = 0.05f;
    DirectTarget guiding_direct_target = DirectTarget::Full;
    std::array<bool, 8> mask_enabled{true, true, true, true, true, true, true, true};

    struct VolumePass {
        Versioned<SlangProgramEntryPoint> entry_point;
        Versioned<Pipeline> pipeline;
        Versioned<ShaderObject> params;
    };
    VolumePass single_scattering;
    VolumePass project_seed;
    VolumePass project;

    SlangCompositionHandle gbuffer_composition;
    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> entry_point;
    Versioned<Pipeline> pipeline;
    Versioned<ShaderBindingTable> sbt;
    Versioned<ShaderObject> params;
};

} // namespace merian
