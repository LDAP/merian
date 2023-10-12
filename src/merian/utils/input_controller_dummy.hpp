#pragma once

#include "merian/utils/input_controller.hpp"

namespace merian {

class DummyInputController : public InputController {

  public:
    explicit DummyInputController();

    ~DummyInputController();

    virtual bool request_raw_mouse_input(bool enable) override;

    // Returns true if raw mouse input is enabled.
    virtual bool get_raw_mouse_input() override;

    // Clear all callbacks
    virtual void reset() override;

    virtual void set_active(bool active) override;

    virtual void set_mouse_cursor_callback(MouseCursorEventCallback cb) override;
    virtual void set_mouse_button_callback(MouseButtonEventCallback cb) override;
    virtual void set_scroll_event_callback(ScrollEventCallback cb) override;
    virtual void set_key_event_callback(KeyEventCallback cb) override;
};

} // namespace merian
