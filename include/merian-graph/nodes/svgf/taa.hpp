#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/nodes/svgf/options.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"

#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

// Blends the filtered irradiance with the previous output and re-modulates the albedo.
class SVGFTAA : public Node {
  private:
    enum class Debug : uint8_t {
        NONE,
        IRRADIANCE,
        VARIANCE,
        NORMAL,
        DEPTH,
        ALBEDO,
        GRAD_Z,
        IRRADIANCE_NAN_INF,
        MV,
    };

    enum class PreviousFilter : uint8_t {
        NONE,
        CATMULL_ROM,
    };

    enum class Clamping : uint8_t {
        MIN_MAX,
        MOMENTS,
    };

    enum class MotionVectorSampling : uint8_t {
        CENTER,
        MAGNITUDE_DILATION,
    };

    struct PushConstant {
        float blend_alpha = 0.0;
        float rejection_threshold = 1.0;
        // the previous output was not written yet
        VkBool32 reset = VK_FALSE;
    };

  public:
    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 Submission& submission) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

    bool modulates_albedo() const;

  private:
    Debug debug = Debug::NONE;
    PreviousFilter filter_prev = PreviousFilter::NONE;
    Clamping clamping = Clamping::MIN_MAX;
    MotionVectorSampling mv_sampling = MotionVectorSampling::CENTER;
    bool enable_mv = true;
    bool modulate_albedo = true;
    PushConstant pc;

    uint32_t local_size = 0;

    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
    const VkSampledImageInHandle con_prev_out = VkSampledImageIn::create();
    GBufferInHandle con_gbuffer;
    // Optional: overrides the gbuffer motion vectors, for signals that are not on the first hit.
    const VkSampledImageInHandle con_mv = VkSampledImageIn::create();

    ManagedVkImageOutHandle con_out;

    SlangProgramEntryPointHandle entry_point;
    PipelineHandle pipeline;
    ShaderObjectHandle globals;
};

} // namespace merian
