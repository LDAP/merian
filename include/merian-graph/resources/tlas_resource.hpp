#pragma once

#include "merian-graph/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

class TLASResource : public GraphResource {
    friend class VkTLASIn;
    friend class VkTLASOut;

  public:
    TLASResource() = default;

    void properties(merian::Properties& props) override {
        if (tlas) {
            tlas->properties(props);
        } else {
            props.output_text("<no TLAS build>");
        }
    }

    TLASResource& operator=(const merian::AccelerationStructureHandle& tlas) {
        this->tlas = tlas;
        return *this;
    }

  private:
    merian::AccelerationStructureHandle tlas;
};

using TLASResourceHandle = std::shared_ptr<TLASResource>;

} // namespace merian
