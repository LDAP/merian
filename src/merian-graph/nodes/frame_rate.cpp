#include "merian-graph/nodes/frame_rate.hpp"

#include <algorithm>

namespace merian {

void FrameRate::properties(Properties& config) {
    config.config_float("framerate", framerate, "", 0.01f);
    framerate = std::max(framerate, 0.01f);
    frametime_millis = 1000.f / framerate;
    if (config.is_ui() && config.config_float("frametime", frametime_millis, "", 0.01f)) {
        frametime_millis = std::max(frametime_millis, 0.01f);
        framerate = 1000.f / frametime_millis;
    }
}

float FrameRate::get_framerate() const {
    return framerate;
}

float FrameRate::get_frametime_millis() const {
    return frametime_millis;
}

std::chrono::nanoseconds FrameRate::frame_time(const uint64_t frame) const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(static_cast<double>(frame) * frametime_millis / 1000.));
}

} // namespace merian
