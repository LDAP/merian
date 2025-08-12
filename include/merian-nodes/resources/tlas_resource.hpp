#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class TLASResource : public GraphResource {
    friend class VkTLASIn;
    friend class VkTLASOut;

  public:
    TLASResource(const vk::PipelineStageFlags2 read_pipeline_stages)
        : input_pipeline_stages(read_pipeline_stages) {}

    void properties(merian::Properties& props) override {
        props.output_text(
            fmt::format("Input pipeline stages: {}", vk::to_string(input_pipeline_stages)));

        if (tlas) {
            tlas->properties(props);
        } else {
            props.output_text("<no TLAS build>");
        }
    }

    TLASResource& operator=(const merian::HWAccelerationStructureHandle& tlas) {
        this->tlas = tlas;
        return *this;
    }

  public:
    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_pipeline_stages;

  private:
    merian::HWAccelerationStructureHandle tlas;
    merian::HWAccelerationStructureHandle last_tlas;
};

using TLASResourceHandle = std::shared_ptr<TLASResource>;

} // namespace merian_nodes
