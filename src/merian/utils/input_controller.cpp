#include "merian/utils/input_controller.hpp"
#include "merian/utils/input_listener.hpp"

#include <algorithm>

namespace merian {

void InputController::add_listener(std::weak_ptr<InputListener> listener, int priority) {
    const auto sp = listener.lock();
    if (!sp)
        return;
    // Prune expired entries and prevent duplicate registration.
    auto it2 = listeners.begin();
    while (it2 != listeners.end()) {
        if (it2->second.expired()) {
            it2 = listeners.erase(it2);
            continue;
        }
        if (it2->second.lock() == sp) {
            listeners.erase(it2);
            break;
        }
        ++it2;
    }
    // Maintain descending sort by priority (highest first).
    auto it = std::lower_bound(listeners.begin(), listeners.end(), priority,
                               [](const auto& pair, int prio) { return pair.first > prio; });
    listeners.insert(it, {priority, std::move(listener)});
}

void InputController::clear_listeners() {
    listeners.clear();
}

void InputController::set_active(bool active) {
    input_active = active;
}

bool InputController::dispatch_cursor(const double xpos, const double ypos) {
    if (!input_active)
        return false;
    auto it = listeners.begin();
    while (it != listeners.end()) {
        auto sp = it->second.lock();
        if (!sp) {
            it = listeners.erase(it);
            continue;
        }
        if (sp->on_cursor(*this, xpos, ypos))
            return true;
        ++it;
    }
    return false;
}

bool InputController::dispatch_mouse_button(const MouseButton button,
                                            const KeyStatus status,
                                            const int mods) {
    if (!input_active)
        return false;
    auto it = listeners.begin();
    while (it != listeners.end()) {
        auto sp = it->second.lock();
        if (!sp) {
            it = listeners.erase(it);
            continue;
        }
        if (sp->on_mouse_button(*this, button, status, mods))
            return true;
        ++it;
    }
    return false;
}

bool InputController::dispatch_scroll(const double xoffset, const double yoffset) {
    if (!input_active)
        return false;
    auto it = listeners.begin();
    while (it != listeners.end()) {
        auto sp = it->second.lock();
        if (!sp) {
            it = listeners.erase(it);
            continue;
        }
        if (sp->on_scroll(*this, xoffset, yoffset))
            return true;
        ++it;
    }
    return false;
}

bool InputController::dispatch_key(const Key key, const KeyStatus action, const int mods) {
    if (!input_active)
        return false;
    auto it = listeners.begin();
    while (it != listeners.end()) {
        auto sp = it->second.lock();
        if (!sp) {
            it = listeners.erase(it);
            continue;
        }
        if (sp->on_key(*this, key, action, mods))
            return true;
        ++it;
    }
    return false;
}

bool InputController::dispatch_char(const unsigned int codepoint) {
    if (!input_active)
        return false;
    auto it = listeners.begin();
    while (it != listeners.end()) {
        auto sp = it->second.lock();
        if (!sp) {
            it = listeners.erase(it);
            continue;
        }
        if (sp->on_char(*this, codepoint))
            return true;
        ++it;
    }
    return false;
}

} // namespace merian
