#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-graph/objects/guiding_object.hpp"
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

// Path tracer starting from the GBuffer hit. Scatter directions come from a one-sample mixture
// of the BSDF and a guiding method plugged into the guiding slot.
class RenderPT : public Node {

  public:
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
    vk::Format irradiance_format = vk::Format::eR32G32B32A32Sfloat;

    void ensure_pipeline(const SceneHandle& scene);
    void update_render_constants();
    void update_guiding_slot();

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    // Connectors
    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();
    ShaderObjectInHandle<GuidingObject> con_guiding = ShaderObjectIn<GuidingObject>::create();
    ShaderObjectInHandle<GuidingObject> con_distance_guiding =
        ShaderObjectIn<GuidingObject>::create();
    ManagedVkImageOutHandle con_irradiance;
    ManagedVkImageOutHandle con_volume;
    ManagedVkImageOutHandle con_volume_depth;
    ManagedVkImageOutHandle con_volume_mv;
    VkSampledImageInHandle con_prev_volume_depth = VkSampledImageIn::create();

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};
    int32_t spp = 1;
    int32_t max_path_length = 5;
    int32_t emitted_max_path_length = max_path_length;
    bool emission_on_primary = true;
    bool enable_ser = false;
    bool use_raygen = true;
    bool demodulate_albedo = false;
    // NullGuidingModel while nothing is connected: the slot then compiles out entirely.
    GuidingModelHandle guiding;
    uint32_t guiding_version = 0;
    GuidingModelHandle distance_guiding;
    uint32_t distance_guiding_version = 0;

    // Single scattering along the primary ray; compiled out where the scene has no medium.
    bool volume_available = false;
    int32_t volume_spp = 1;
    int32_t volume_nee_candidates = 0;
    bool volume_forward_project = true;
    float volume_forward_project_min_z = 50.f;
    vk::Format volume_depth_format = vk::Format::eR32Sfloat;

    // 0 = one draw from the mixture (MIS), 1 = resample several (RIS)
    int32_t scatter_mode = 0;
    int32_t scatter_candidates = 2;

    int32_t nee_mode = 2;
    float nee_probability = 0.1f;
    int32_t nee_candidates = 1;
    int32_t nee_bounces = 0;
    std::array<bool, 8> mask_enabled{true, true, true, true, true, true, true, true};

    struct VolumePass {
        Versioned<SlangProgramEntryPoint> entry_point;
        Versioned<Pipeline> pipeline;
        Versioned<ShaderObject> params;
    };
    VolumePass single_scattering;
    VolumePass project_seed;
    VolumePass project;

    // Slang program + pipeline; rebuilt when the scene composition changes.
    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> entry_point;
    Versioned<Pipeline> pipeline;
    Versioned<ShaderBindingTable> sbt;
    Versioned<ShaderObject> params;
};

} // namespace merian
