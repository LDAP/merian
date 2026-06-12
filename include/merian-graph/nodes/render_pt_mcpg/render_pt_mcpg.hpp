#pragma once

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-shaders/gbuffer.hpp"
#include "merian-scene/scene.hpp"

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

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    // Connectors
    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    PtrInHandle<GBuffer> con_gbuffer = PtrIn<GBuffer>::create();
    ManagedVkImageOutHandle con_irradiance;

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};
    int32_t spp = 1;
    int32_t max_path_length = 8;
    int32_t emitted_max_path_length = max_path_length;
    bool emission_on_primary = false;
    std::array<bool, 8> mask_enabled{true, true, true, true, true, true, true, true};

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
