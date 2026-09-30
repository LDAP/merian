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
    // match SCATTER_MODE_*, NEE_MODE_* and OUTPUT_VIEW_* in the shaders
    enum class ScatterMode : int32_t { MIS, RIS };
    enum class NEEMode : int32_t { Off, Mixture, Resampled };
    enum class TraceShader : int32_t { Auto, RayGeneration, Compute };
    enum class OutputView : int32_t { Radiance, GuidingDebug, ScatterStatistics };

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
    void ensure_pipeline(const SceneHandle& scene);
    void update_render_constants();
    uint32_t recorded_vertices_per_path() const;
    bool use_raygen() const;
    void update_guiding_slot();
    bool has_guiding() const;
    float nee_probability() const;
    float guided_probability() const;

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    GBufferInHandle con_gbuffer;
    ShaderObjectInHandle<GuidingObject> con_guiding = ShaderObjectIn<GuidingObject>::create();
    ShaderObjectInHandle<GuidingObject> con_distance_guiding =
        ShaderObjectIn<GuidingObject>::create();
    ManagedVkImageOutHandle con_irradiance;
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
    int32_t emitted_max_path_length = max_path_length;
    bool emission_on_primary = true;
    bool enable_ser = false;
    TraceShader trace_shader = TraceShader::Auto;
    bool raygen_preferred = true;
    bool russian_roulette = true;
    bool demodulate_albedo = false;
    GuidingModelHandle guiding;
    uint32_t guiding_version = 0;
    GuidingModelHandle distance_guiding;
    uint32_t distance_guiding_version = 0;

    bool volume_available = false;
    int32_t volume_spp = 1;
    bool volume_forward_project = true;
    float volume_forward_project_min_z = 50.f;
    vk::Format volume_depth_format = vk::Format::eR32Sfloat;

    OutputView output_view = OutputView::Radiance;
    bool follow_specular = true;
    float specular_alpha = 0.f;
    ScatterMode scatter_mode = ScatterMode::MIS;
    int32_t scatter_candidates = 2;

    float bsdf_share = 0.45f;
    float nee_share = 0.1f;
    float guiding_share = 0.45f;

    NEEMode nee_mode = NEEMode::Resampled;
    int32_t nee_bounces = 0;

    bool guiding_scale_with_alpha = true;
    float guiding_alpha_threshold = 0.05f;
    int32_t guiding_direct_target = 0;
    float guiding_distance_share = 0.9f;
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
