#include "merian/vk/window/window.hpp"

namespace merian {

void Window::add_window_listener(std::weak_ptr<WindowListener> listener) {
    if (auto p = listener.lock()) {
        p->on_resize(framebuffer_extent(), window_extent());
        p->on_display_scale_changed(get_display_scale());
    }
    window_listeners.push_back(std::move(listener));
}

void Window::clear_window_listeners() {
    window_listeners.clear();
}

void Window::dispatch_resize(vk::Extent2D framebuffer_extent, vk::Extent2D window_extent) {
    for (auto it = window_listeners.begin(); it != window_listeners.end();) {
        if (auto p = it->lock()) {
            p->on_resize(framebuffer_extent, window_extent);
            ++it;
        } else {
            it = window_listeners.erase(it);
        }
    }
}

void Window::dispatch_display_scale_changed(float display_scale) {
    for (auto it = window_listeners.begin(); it != window_listeners.end();) {
        if (auto p = it->lock()) {
            p->on_display_scale_changed(display_scale);
            ++it;
        } else {
            it = window_listeners.erase(it);
        }
    }
}

void Window::dispatch_focus_changed(bool focused) {
    for (auto it = window_listeners.begin(); it != window_listeners.end();) {
        if (auto p = it->lock()) {
            p->on_focus_changed(focused);
            ++it;
        } else {
            it = window_listeners.erase(it);
        }
    }
}

void Window::dispatch_minimized() {
    for (auto it = window_listeners.begin(); it != window_listeners.end();) {
        if (auto p = it->lock()) {
            p->on_minimized();
            ++it;
        } else {
            it = window_listeners.erase(it);
        }
    }
}

void Window::dispatch_restored() {
    for (auto it = window_listeners.begin(); it != window_listeners.end();) {
        if (auto p = it->lock()) {
            p->on_restored();
            ++it;
        } else {
            it = window_listeners.erase(it);
        }
    }
}

} // namespace merian
