#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"

#include "merian/shader/slang_entry_point.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

#include <optional>

namespace merian {

class SVGF : public Node {
  private:
    struct VarianceEstimatePushConstant {
        float normal_reject_cos = 0.8;
        float depth_accept = 10; // larger reuses more
        float spatial_falloff = 3.0;
        float spatial_bias = 8.0;
    };

    struct FilterPushConstant {
        float param_z = 10; // parameter for depth      = 1   larger blurs more
        float param_n = .8; // parameter for normals    cos(alpha) for lower threshold
        float param_l = 8;  // parameter for brightness = 4   larger blurs more
        float z_bias_normals = -1.0;
        float z_bias_depth = -1.0;
    };

    struct TAAPushConstant {
        float blend_alpha = 0.0;
        float rejection_threshold = 1.0;
    };

  public:
    SVGF();

    ~SVGF();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeConnectedInfo& info) override;

    void process(GraphRun& run, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    std::optional<vk::Format> output_format = std::nullopt;

    // depends on available shared memory
    uint32_t variance_estimate_local_size;
    uint32_t filter_local_size;
    uint32_t taa_local_size;

    VkSampledImageInHandle con_prev_out = VkSampledImageIn::create();
    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    VkSampledImageInHandle con_history = VkSampledImageIn::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();

    ManagedVkImageOutHandle con_out;

    std::shared_ptr<SlangProgramEntryPoint> variance_estimate_module;
    std::shared_ptr<SlangProgramEntryPoint> filter_module;
    std::shared_ptr<SlangProgramEntryPoint> taa_module;

    ShaderObjectHandle variance_estimate_globals;
    // [d] reads ping_pong_res[d], writes ping_pong_res[d ^ 1]
    std::array<ShaderObjectHandle, 2> filter_globals;
    ShaderObjectHandle taa_globals;

    VarianceEstimatePushConstant variance_estimate_pc;
    FilterPushConstant filter_pc;
    TAAPushConstant taa_pc;

    vk::ImageCreateInfo irr_create_info;

    PipelineHandle variance_estimate;
    std::vector<PipelineHandle> filters;
    PipelineHandle taa;

    int svgf_iterations = 0;

    struct EAWRes {
        TextureHandle ping_pong;
        TextureHandle gbuf_ping_pong;
    };
    std::array<EAWRes, 2> ping_pong_res; // Ping pong textures

    int filter_type = 2;

    int taa_debug = 0;
    int taa_filter_prev = 0;
    int taa_clamping = 0;
    int taa_mv_sampling = 0;
    bool enable_mv = true;
    bool taa_modulate_albedo = true;

    bool kaleidoscope = false;
    bool kaleidoscope_use_shmem = true;
};

} // namespace merian
