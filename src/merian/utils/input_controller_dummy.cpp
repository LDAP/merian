#include "merian/utils/input_controller_dummy.hpp"

namespace merian {

DummyInputController::DummyInputController() {}

DummyInputController::~DummyInputController() {}

bool DummyInputController::request_raw_mouse_input(bool) {
    return false;
}

bool DummyInputController::get_raw_mouse_input() {
    return false;
}

// Clear all callbacks
void DummyInputController::reset() {}

void DummyInputController::set_active(bool) {}

void DummyInputController::set_mouse_cursor_callback(MouseCursorEventCallback) {}
void DummyInputController::set_mouse_button_callback(MouseButtonEventCallback) {}
void DummyInputController::set_scroll_event_callback(ScrollEventCallback) {}
void DummyInputController::set_key_event_callback(KeyEventCallback) {}

} // namespace merian
