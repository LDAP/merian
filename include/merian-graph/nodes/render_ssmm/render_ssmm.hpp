#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing.hpp"
#include "merian/vk/raytrace/shader_binding_table.hpp"
#include <optional>

namespace merian {

// Renders a scene using screen-space mixture models by Dittebrandt et al. (2023).
class RenderSSMM : public Node {

  public:
    RenderSSMM();

    ~RenderSSMM() override = default;

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

    vk::BufferCreateInfo ssmc_buffer_create_info() const;
    void update_render_constants();

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    // Connectors
    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();
    VkBufferInHandle con_prev_ssmc = VkBufferIn::create();
    ManagedVkImageOutHandle con_irradiance;
    ManagedVkBufferOutHandle con_ssmc;

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};

    int32_t spp = 1;
    float surf_bsdf_p = 0.15;

    float ml_prior_n = .20;
    uint32_t ml_max_n = 1024;
    float ml_min_alpha = 0.01;
    uint32_t smis_group_size = 5;

    uint32_t seed = 0;
    bool randomize_seed = true;

    bool ssmc_needs_reset = true;

    // Slang program + pipeline; rebuilt when the scene composition changes.
    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> entry_point;
    Versioned<RayTracingPipeline> pipeline;
    Versioned<ShaderBindingTable> sbt;
    Versioned<ShaderObject> params;
    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;
};

} // namespace merian
