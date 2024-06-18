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
        tlas->properties(props);
    }

  private:
    merian::AccelerationStructureHandle tlas;
    std::vector<merian::AccelerationStructureHandle> in_flight_tlas;

    bool needs_descriptor_update = true;
};

using TLASResourceHandle = std::shared_ptr<TLASResource>;

} // namespace merian_nodes
