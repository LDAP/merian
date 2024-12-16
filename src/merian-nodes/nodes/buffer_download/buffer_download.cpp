#include "merian-nodes/nodes/buffer_download/buffer_download.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian-nodes/graph/errors.hpp"

#include <iostream>

namespace merian_nodes {

BufferDownload::BufferDownload(const ContextHandle context) : Node(), context(context) {}

BufferDownload::~BufferDownload() {}

std::vector<InputConnectorHandle> BufferDownload::describe_inputs() {

    return {con_src};
}

std::vector<OutputConnectorHandle> BufferDownload::describe_outputs(const NodeIOLayout& io_layout) {
    return {};
}

BufferDownload::NodeStatusFlags
BufferDownload::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                           const DescriptorSetLayoutHandle& descriptor_set_layout) {
    return {};
}

void BufferDownload::process([[maybe_unused]] GraphRun& run,
                           const vk::CommandBuffer& cmd,
                           const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {

    if (results.size() != run.get_iterations_in_flight()) {
        results.resize(run.get_iterations_in_flight());
    }

    uint32_t curr_idx = run.get_in_flight_index();
    if (results[curr_idx] == nullptr) {
        results[curr_idx] = static_cast<const glm::vec4*>(run.get_allocator()->getStaging()->cmdFromBuffer(cmd, **io[con_src], 0, sizeof(glm::vec4)));
    }

    if (results[curr_idx] != nullptr) {
        std::cout << results[curr_idx][0].x << " " << results[curr_idx][0].y << " " << results[curr_idx][0].z << " " << results[curr_idx][0].w << std::endl;
    }
}

} // namespace merian_nodes
