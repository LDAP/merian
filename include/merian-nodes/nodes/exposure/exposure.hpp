#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"
#include "merian-nodes/connectors/vk_texture_in.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian_nodes {

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
    AutoExposure(const ContextHandle& context);

    virtual ~AutoExposure();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const ContextHandle context;

    VkTextureInHandle con_src = VkTextureIn::compute_read("src");

    ManagedVkImageOutHandle con_out;
    ManagedVkBufferOutHandle con_hist;
    ManagedVkBufferOutHandle con_luminance;

    PushConstant pc;

    ShaderModuleHandle histogram_module;
    ShaderModuleHandle luminance_module;
    ShaderModuleHandle exposure_module;

    PipelineHandle histogram;
    PipelineHandle luminance;
    PipelineHandle exposure;
};

} // namespace merian_nodes
