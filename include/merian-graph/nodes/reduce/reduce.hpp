#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/nodes/compute_node/compute_node.hpp"

namespace merian {

class Reduce : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  public:
    Reduce();

    ~Reduce();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    SlangCompositionHandle create_composition() override;

    NodeStatusFlags properties(Properties& props) override;

  private:
    std::string generate_source() const;

    bool try_compile(const std::string& source_candidate);

    std::optional<vk::Format> output_format = std::nullopt;

    // Unique per instance: several Reduce nodes coexist (e.g. compositing and summing), and the
    // session caches generated modules by name.
    const std::string module_name;

    std::string source;
    std::optional<std::string> error;

    std::string initial_value = "vec4(0)";
    std::string reduction = "accumulator + current_value";

    vk::Extent3D extent;

    uint32_t number_inputs = 10;
    std::vector<VkSampledImageInHandle> input_connectors;
    std::vector<uint32_t> connected_indices;
};

} // namespace merian
