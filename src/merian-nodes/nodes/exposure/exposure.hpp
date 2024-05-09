#pragma once

#include "merian/utils/stopwatch.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class ExposureNode : public Node {
  private:
    // Histogram uses local_size_x * local_size_y bins;
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        int automatic = false;

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
    ExposureNode(const SharedContext context, const ResourceAllocatorHandle allocator);

    virtual ~ExposureNode();

    virtual std::string name() override;

    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    virtual std::tuple<std::vector<NodeOutputDescriptorImage>,
                       std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(
        const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
        const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) override;

    virtual void cmd_build(const vk::CommandBuffer& cmd, const std::vector<NodeIO>& ios) override;

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             GraphRun& run,
                             const std::shared_ptr<FrameData>& frame_data,
                             const uint32_t set_index,
                             const NodeIO& io) override;

    virtual void get_configuration(Configuration& config, bool& needs_rebuild) override;

  private:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;

    PushConstant pc;

    std::vector<TextureHandle> graph_textures;
    std::vector<DescriptorSetHandle> graph_sets;
    DescriptorSetLayoutHandle graph_layout;
    DescriptorPoolHandle graph_pool;

    ShaderModuleHandle histogram_module;
    ShaderModuleHandle luminance_module;
    ShaderModuleHandle exposure_module;

    PipelineHandle histogram;
    PipelineHandle luminance;
    PipelineHandle exposure;

    Stopwatch sw;
};

} // namespace merian
