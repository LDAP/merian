#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"

#include "merian/utils/enums.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/imgui/imgui_merian_backend.hpp"
#include "merian/vk/imgui/imgui_renderer.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"

#include <array>
#include <deque>
#include <fstream>
#include <mutex>
#include <optional>

namespace merian {

enum class ErrorMetric : uint32_t {
    MSE,
    RMSE,
    MAE,
    RelMSE,
};

static constexpr std::array<ErrorMetric, 4> ERROR_METRIC_VALUES = {
    ErrorMetric::MSE, ErrorMetric::RMSE, ErrorMetric::MAE, ErrorMetric::RelMSE};

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
    case ErrorMetric::RelMSE:
        return "relative MSE";
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
        uint32_t relative;
        float epsilon;
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
                                 const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 Submission& submission) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    // Per-channel error of the latest reduction, already converted for the selected metric.
    float4 metric_error() const;
    // The reference is another node's output, an image on disk, or a snapshot of the input.
    void load_reference(const NodeIO& io, Submission& submission);
    void take_snapshot(const NodeIO& io, Submission& submission);
    void reset_history();
    void record_output(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);

    std::string reference_path;
    bool reference_dirty = false;
    bool reference_owned = false; // a file or snapshot stands in for the connected image
    bool reference_is_snapshot = false;
    bool snapshot_requested = false;
    TextureHandle reference_texture;
    // measurements dispatched since the last reset
    uint32_t submitted = 0;
    // bumped on reset so readbacks against the previous reference are dropped
    uint32_t generation = 0;
    // floor under the reference's magnitude, below which a black pixel cannot dominate relMSE
    float relative_epsilon = 1e-2f;
    std::string csv_path;
    std::ofstream csv_stream;

    void draw_overlay(uint32_t width, uint32_t height) const;

    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;

    // both inputs are sampled by the metric shader and blitted into the split view
    const VkSampledImageInHandle con_reference = VkSampledImageIn::create();
    const VkSampledImageInHandle con_input = VkSampledImageIn::create();
    ManagedVkImageOutHandle con_out;
    ManagedVkBufferOutHandle con_error;

    PushConstant pc{};

    Versioned<SpecializationInfo> error_to_buffer_spec;
    Versioned<SpecializationInfo> reduce_buffer_spec;

    std::optional<ComputeKernel> error_to_buffer_kernel;
    std::optional<ComputeKernel> reduce_buffer_kernel;

    // overlay rendering onto the node output
    ImGuiContextHandle imgui_ctx;
    ImGuiRendererHandle imgui_renderer;
    ImGuiMerianBackendHandle imgui_backend;
    Stopwatch frametime;

    // host-visible readback, one buffer per in-flight iteration so a slot is only reused once its
    // sync_to_cpu callback has run
    std::vector<BufferHandle> readback_buffers;

    struct Readback {
        uint32_t generation;
        uint32_t sample;
        float4 sum; // raw per-channel mean error
    };
    std::mutex result_mutex;
    std::vector<Readback> readbacks;

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
