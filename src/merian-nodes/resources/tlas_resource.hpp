#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class TLASResource : public GraphResource {
    friend class VkTLASIn;
    friend class VkTLASOut;

  public:
    TLASResource(const vk::PipelineStageFlags2 read_pipeline_stages, const uint32_t ring_size)
        : input_pipeline_stages(read_pipeline_stages), in_flight_tlas(ring_size) {}

    void properties(merian::Properties& props) override {
        props.output_text(
            fmt::format("Input pipeline stages: {}", vk::to_string(input_pipeline_stages)));

        if (tlas) {
            tlas->properties(props);
        } else {
            props.output_text("<no TLAS build>");
        }
    }

    void operator=(const merian::AccelerationStructureHandle& tlas) {
        this->tlas = tlas;
    }

  public:
    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_pipeline_stages;

  private:
    merian::AccelerationStructureHandle tlas;
    merian::AccelerationStructureHandle last_tlas;

    std::vector<merian::AccelerationStructureHandle> in_flight_tlas;
};

using TLASResourceHandle = std::shared_ptr<TLASResource>;

} // namespace merian_nodes
