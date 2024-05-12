#pragma once

#include "connector.hpp"

namespace merian {

class InputConnector : public Connector {
  public:
    // Returns the number of iterations the corresponing resource is accessed later.
    virtual uint32_t get_delay() const {
        return 0;
    }
};

} // namespace merian
