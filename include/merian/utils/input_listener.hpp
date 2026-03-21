#pragma once

#include "merian/utils/input_controller.hpp"

#include <memory>

namespace merian {

// A listener that receives input events dispatched by an InputController.
//
// Listeners are registered on a controller via add_listener(). The controller
// holds weak_ptrs, so listeners are removed automatically when destroyed.
//
// Return true from an on_xxx handler to consume the event and prevent lower-
// priority listeners from seeing it. Return false to let it pass through.
class InputListener {
  public:
    virtual ~InputListener() = default;

    virtual bool on_cursor(InputController& /*controller*/, double /*xpos*/, double /*ypos*/) {
        return false;
    }
    virtual bool on_mouse_button(InputController& /*controller*/,
                                 InputController::MouseButton /*button*/,
                                 InputController::KeyStatus /*status*/,
                                 int /*mods*/) {
        return false;
    }
    virtual bool
    on_scroll(InputController& /*controller*/, double /*xoffset*/, double /*yoffset*/) {
        return false;
    }
    virtual bool on_key(InputController& /*controller*/,
                        InputController::Key /*key*/,
                        InputController::KeyStatus /*action*/,
                        int /*mods*/) {
        return false;
    }
    virtual bool on_char(InputController& /*controller*/, unsigned int /*codepoint*/) {
        return false;
    }
};

using InputListenerHandle = std::shared_ptr<InputListener>;

} // namespace merian
