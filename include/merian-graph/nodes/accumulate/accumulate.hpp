#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"

#include <optional>

namespace merian {

class Accumulate : public Node {
  private:
    struct QuartilePushConstant {
        float firefly_percentile_lower = 0.25;
        float firefly_percentile_upper = 0.90;

        float adaptive_alpha_percentile_lower = 0.05;
        float adaptive_alpha_percentile_upper = 0.95;
    };

    struct FilterPushConstant {
        VkBool32 firefly_filter_enable = 0;
        float firefly_bias = 0.2;
        float firefly_ipr_factor = 50;

        float firefly_hard_clamp = INFINITY;

        float accum_alpha = 0.0;
        float accum_max_hist = INFINITY;
        float normal_reject_cos = 0.8;
        float depth_reject_percent = 0.02;
        VkBool32 clear = 0;

        float adaptive_alpha_reduction = 0.0;
        float adaptive_alpha_ipr_factor = 1.5;

        uint32_t iteration = 0;
    };

  public:
    Accumulate();

    ~Accumulate();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeConnectedInfo& info) override;

    void process(GraphRun& run, const NodeIO& io) override;

    NodeStatusFlags properties([[maybe_unused]] Properties& config) override;

    // Clears the accumulation buffer at the next iteration.
    void request_clear();

  private:
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;
    std::optional<vk::Format> format = vk::Format::eR32G32B32A32Sfloat;

    static constexpr uint32_t PERCENTILE_LOCAL_SIZE_X = 8;
    static constexpr uint32_t PERCENTILE_LOCAL_SIZE_Y = 8;
    static constexpr uint32_t FILTER_LOCAL_SIZE_X = 16;
    static constexpr uint32_t FILTER_LOCAL_SIZE_Y = 16;

    // Graph IO
    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();
    VkSampledImageInHandle con_prev_out = VkSampledImageIn::create();
    ShaderObjectInHandle<GBufferObject> con_prev_gbuffer = ShaderObjectIn<GBufferObject>::create();
    VkSampledImageInHandle con_prev_history = VkSampledImageIn::create();

    ManagedVkImageOutHandle con_out;
    ManagedVkImageOutHandle con_history;

    vk::ImageCreateInfo irr_create_info;

    uint32_t percentile_group_count_x;
    uint32_t percentile_group_count_y;
    uint32_t filter_group_count_x;
    uint32_t filter_group_count_y;

    TextureHandle percentile_texture;

    Versioned<SpecializationInfo> percentile_spec_info;
    Versioned<SpecializationInfo> accumulate_spec_info;
    std::optional<ComputeKernel> percentile_kernel;
    std::optional<ComputeKernel> accumulate_kernel;

    FilterPushConstant accumulate_pc;
    QuartilePushConstant percentile_pc;

    bool clear = false;
    int filter_mode = 0;
    VkBool32 extended_search = VK_TRUE;
    VkBool32 reuse_border = VK_FALSE;
    bool enable_mv = VK_TRUE;
    int gbuffer_check_mode = 0;

    std::string clear_event_listener_pattern = "/user/clear";
};

} // namespace merian
