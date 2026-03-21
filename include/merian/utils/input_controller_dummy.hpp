#pragma once

#include "merian/utils/input_controller.hpp"

namespace merian {

class DummyInputController : public InputController {

  public:
    explicit DummyInputController() = default;
    ~DummyInputController() = default;

};

} // namespace merian
