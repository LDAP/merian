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
#include "merian/vk/pipeline/pipeline_ray_tracing.hpp"
#include "merian/vk/raytrace/shader_binding_table.hpp"

#include <array>

namespace merian {

class RenderRestirDI : public Node {

  public:
    enum Pass { Generate = 0, Temporal = 1, Spatial = 2, Shade = 3, PassCount = 4 };

    RenderRestirDI();

    ~RenderRestirDI() override = default;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeConnectedInfo& info) override;

    void process(GraphRun& run, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    vk::BufferCreateInfo reservoir_buffer_create_info() const;
    void update_render_constants();

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();
    ShaderObjectInHandle<GBufferObject> con_prev_gbuffer = ShaderObjectIn<GBufferObject>::create();
    VkBufferInHandle con_prev_reservoirs = VkBufferIn::create();
    ManagedVkImageOutHandle con_irradiance;
    ManagedVkBufferOutHandle con_reservoirs;

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};

    int32_t spp = 4;
    uint32_t seed = 0;
    bool emission_on_primary = true;
    bool demodulate_albedo = false;

    bool temporal_enable = true;
    float temporal_normal_reject_cos = 0.96f;
    float temporal_depth_reject = 0.1f;
    int32_t temporal_clamp_m = 32 * 20;
    int32_t temporal_bias_correction = 2;
    bool apply_mv = false;
    float boiling_filter_strength = 0.0f;

    int32_t spatial_iterations = 1;
    float spatial_normal_reject_cos = 0.96f;
    float spatial_depth_reject = 0.1f;
    int32_t spatial_radius = 30;
    int32_t spatial_bias_correction = 1;

    bool visibility_shade = true;

    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    std::array<Versioned<SlangProgramEntryPoint>, PassCount> entry_points;
    std::array<Versioned<RayTracingPipeline>, PassCount> pipelines;
    std::array<Versioned<ShaderBindingTable>, PassCount> sbts;
    std::array<Versioned<ShaderObject>, PassCount> params;
    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;

    BufferHandle pong_buffer;
};

} // namespace merian
