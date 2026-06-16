#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node.hpp"

#include "merian/shader/entry_point.hpp"
#include "merian/utils/enums.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/imgui/imgui_merian_backend.hpp"
#include "merian/vk/imgui/imgui_renderer.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

#include <array>
#include <deque>
#include <mutex>

namespace merian {

enum class ErrorMetric : uint32_t {
    MSE,
    RMSE,
    MAE,
};

static constexpr std::array<ErrorMetric, 3> ERROR_METRIC_VALUES = {
    ErrorMetric::MSE, ErrorMetric::RMSE, ErrorMetric::MAE};

template <> inline uint32_t enum_size<ErrorMetric>() {
    return ERROR_METRIC_VALUES.size();
}
template <> inline const ErrorMetric* enum_values<ErrorMetric>() {
    return ERROR_METRIC_VALUES.data();
}
template <> inline std::string enum_to_string<ErrorMetric>(const ErrorMetric value) {
    switch (value) {
    case ErrorMetric::MSE:
        return "MSE";
    case ErrorMetric::RMSE:
        return "RMSE";
    case ErrorMetric::MAE:
        return "MAE";
    }
    return "unknown";
}

// Computes a per-pixel error of an input image against a reference and reduces it to a single
// value on the GPU. The result is read back asynchronously and shown as an error-over-time plot
// overlaid on a split view of reference (left) and input (right).
class ErrorPlot : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;
    static constexpr uint32_t workgroup_size = local_size_x * local_size_y;

    struct PushConstant {
        uint32_t divisor;

        int32_t size;
        int32_t offset;
        int32_t count;

        uint32_t squared;
    };

  public:
    ErrorPlot();

    ~ErrorPlot();

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
    // Per-channel error of the latest reduction, already converted for the selected metric.
    float4 metric_error() const;

    void draw_overlay(uint32_t width, uint32_t height) const;

    ContextHandle context;
    ResourceAllocatorHandle allocator;

    // both inputs are sampled by the metric shader and blitted into the split view
    const VkSampledImageInHandle con_reference = std::make_shared<VkSampledImageIn>(
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
        vk::ShaderStageFlagBits::eCompute);
    const VkSampledImageInHandle con_input = std::make_shared<VkSampledImageIn>(
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
        vk::ShaderStageFlagBits::eCompute);
    ManagedVkImageOutHandle con_out;
    ManagedVkBufferOutHandle con_error;

    PushConstant pc{};

    EntryPointHandle error_to_buffer_shader;
    EntryPointHandle reduce_buffer_shader;
    PipelineHandle error_to_buffer;
    PipelineHandle reduce_buffer;

    // overlay rendering onto the node output
    ImGuiContextHandle imgui_ctx;
    ImGuiRendererHandle imgui_renderer;
    ImGuiMerianBackendHandle imgui_backend;
    Stopwatch frametime;

    // host-visible readback, one buffer per in-flight iteration so a slot is only reused once its
    // sync_to_cpu callback has run
    std::vector<BufferHandle> readback_buffers;

    std::mutex result_mutex;
    float4 latest_sum{}; // raw per-channel mean error, written by the readback callback
    bool latest_valid = false;

    // graph-thread copy of the latest readback, safe to read without locking
    float4 current_sum{};

    // config
    ErrorMetric metric = ErrorMetric::RMSE;
    uint32_t history_size = 256;
    std::deque<float> history;

    bool log_x_axis = false;
    bool log_y_axis = false;

    bool auto_scale = true;
    float scale_min = 0.0f;
    float scale_max = 1.0f;
};

} // namespace merian
