#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class TLASResource : public GraphResource {
    friend class VkTLASIn;
    friend class VkTLASOut;

  public:
    TLASResource(const uint32_t ring_size) : in_flight_tlas(ring_size) {}

    void properties(merian::Properties& props) override {
        if (tlas) {
            tlas->properties(props);
        } else {
            props.output_text("<no TLAS build>");
        }
    }

  private:
    merian::AccelerationStructureHandle tlas;
    merian::AccelerationStructureHandle last_tlas;

    std::vector<merian::AccelerationStructureHandle> in_flight_tlas;
};

using TLASResourceHandle = std::shared_ptr<TLASResource>;

} // namespace merian_nodes
