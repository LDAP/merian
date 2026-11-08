#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"

#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"

#include "merian/vk/pipeline/specialization_info.hpp"

#include <optional>

namespace merian {

class Bloom : public Node {
  private:
    // Undefined: keep the format of the input.
    vk::Format overwrite_format = vk::Format::eUndefined;

    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;
    static constexpr uint32_t max_levels = 12;

    struct PushConstant {
        int32_t level = 0;
        int32_t coarse_is_down = 0;
        int32_t karis = 0;
        float fine_weight = 1.0;
        float coarse_weight = 1.0;
        float intensity = 0.04;
        float radius = 1.0;
    };

  public:
    Bloom();

    virtual ~Bloom();

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
    std::vector<float> level_weights(uint32_t level_count) const;

    void bind_mip_chain();

    void barrier_written(const CommandBufferHandle& cmd, const ImageHandle& image);

    ShaderCompileContextHandle compile_context;

    VkSampledImageInHandle con_src;
    ManagedVkImageOutHandle con_out;

    PushConstant pc;
    Versioned<SpecializationInfo> spec_info;

    std::optional<ComputeKernel> downsample_kernel;
    std::optional<ComputeKernel> upsample_kernel;
    std::optional<ComputeKernel> composite_kernel;

    ImageHandle down_image;
    ImageHandle up_image;
    std::vector<TextureHandle> down_levels;
    std::vector<TextureHandle> up_levels;
    uint32_t chain_level_count = 0;
    uint32_t image_height = 0;

    float size = 0.2;
    float core = 0.001;
    float falloff = 2.5;
    bool karis_average = false;
    int32_t mode = 0;
};

} // namespace merian
