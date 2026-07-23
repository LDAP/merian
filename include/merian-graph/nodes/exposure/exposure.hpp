#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"

#include "merian/vk/pipeline/specialization_info.hpp"

#include <optional>

namespace merian {

class AutoExposure : public Node {
  private:
    // Histogram uses local_size_x * local_size_y bins;
    static constexpr uint32_t LOCAL_SIZE_X = 16;
    static constexpr uint32_t LOCAL_SIZE_Y = 16;

    struct PushConstant {
        int automatic = VK_FALSE;

        float iso = 100.0;
        float q = 0.65;

        // Manual exposure
        float shutter_time = .1;
        float aperature = 16.0;

        // Auto exposure
        float K = 8.0;
        float speed_up = 3.0;
        float speed_down = 5.0;
        float timediff = 0;
        int reset = 0;
        float min_log_histogram = -15.0;
        float max_log_histogram = 11.0;
        int metering = 1;
        float min_exposure = 1;
        float max_exposure = 1e9;
    };

  public:
    AutoExposure();

    virtual ~AutoExposure();

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
    ShaderCompileContextHandle compile_context;

    VkSampledImageInHandle con_src = VkSampledImageIn::create();

    ManagedVkImageOutHandle con_out;
    ManagedVkBufferOutHandle con_hist;
    ManagedVkBufferOutHandle con_luminance;

    PushConstant pc;
    Versioned<SpecializationInfo> spec_info;

    std::optional<ComputeKernel> histogram_kernel;
    std::optional<ComputeKernel> luminance_kernel;
    std::optional<ComputeKernel> exposure_kernel;
};

} // namespace merian
