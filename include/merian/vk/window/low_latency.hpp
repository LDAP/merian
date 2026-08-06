#pragma once

#include "merian/utils/enums.hpp"
#include "merian/utils/properties.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/window/swapchain.hpp"

#include <array>
#include <chrono>

namespace merian {

enum class LowLatencyMode : uint32_t {
    OFF,
    AUTO,
    NV_REFLEX,
    AMD_ANTI_LAG,
};

static constexpr std::array<LowLatencyMode, 4> LOW_LATENCY_MODE_VALUES = {
    LowLatencyMode::OFF, LowLatencyMode::AUTO, LowLatencyMode::NV_REFLEX,
    LowLatencyMode::AMD_ANTI_LAG};

template <> inline uint32_t enum_size<LowLatencyMode>() {
    return LOW_LATENCY_MODE_VALUES.size();
}
template <> inline const LowLatencyMode* enum_values<LowLatencyMode>() {
    return LOW_LATENCY_MODE_VALUES.data();
}
template <> inline std::string enum_to_string<LowLatencyMode>(const LowLatencyMode value) {
    switch (value) {
    case LowLatencyMode::OFF:
        return "off";
    case LowLatencyMode::AUTO:
        return "auto";
    case LowLatencyMode::NV_REFLEX:
        return "NV Reflex";
    case LowLatencyMode::AMD_ANTI_LAG:
        return "AMD Anti-Lag";
    }
    return "unknown";
}

/**
 * @brief Delays the start of a frame until just in time, to reduce input-to-photon latency.
 *
 * Backed by VK_NV_low_latency2 (Reflex) or VK_AMD_anti_lag, whichever the device supports. Both
 * pace the frame against the presentation queue and therefore need a swapchain; without one the
 * mode stays inactive.
 *
 * Drive it from the frame loop: begin_frame() before input and simulation, begin_render() before
 * commands are recorded. The present-side work is installed into the swapchain via its hooks.
 */
class LowLatency {
  public:
    LowLatency(const ContextHandle& context);

    ~LowLatency();

    LowLatency(const LowLatency&) = delete;
    LowLatency& operator=(const LowLatency&) = delete;

    // The swapchain that presents the frames. Must be set before its first acquire, otherwise the
    // swapchain is created without the structures Reflex needs.
    void set_swapchain(const SwapchainHandle& swapchain);

    // Blocks until the frame should start.
    void begin_frame();

    // The frame is done simulating and starts recording and submitting commands.
    void begin_render();

    void properties(Properties& props);

  private:
    // What the current mode resolves to, given availability and the attached swapchain.
    enum class Backend {
        NONE,
        NV_REFLEX,
        AMD_ANTI_LAG,
    };

    Backend resolve_backend() const;

    // Applies backend switches and parameter changes. Returns the swapchain Reflex operates on,
    // or VK_NULL_HANDLE.
    vk::SwapchainKHR update_backend();

    void set_marker(const vk::SwapchainKHR& swapchain, const vk::LatencyMarkerNV marker);

    void anti_lag_update(const vk::AntiLagStageAMD stage);

    void on_present_begin();

    void on_present_end();

  private:
    const ContextHandle context;
    const bool nv_reflex_supported;
    const bool amd_anti_lag_supported;

    LowLatencyMode mode = LowLatencyMode::OFF;
    // NV: renders at the highest clocks even when the GPU is not saturated.
    bool boost = false;
    // Frame rate cap the backend paces towards. 0 means uncapped.
    uint32_t max_fps = 0;

    std::weak_ptr<Swapchain> swapchain;

    // The state the backend was last configured with, to detect what needs to be re-applied.
    Backend applied_backend = Backend::NONE;
    vk::SwapchainKHR applied_swapchain = VK_NULL_HANDLE;
    bool applied_boost = false;
    uint32_t applied_max_fps = 0;

    TimelineSemaphoreHandle sleep_semaphore;
    uint64_t sleep_value = 0;

    // Identifies the frame across the markers and the present. Strictly increasing.
    uint64_t present_id = 0;
    // Chained into the swapchain create info / the present; must outlive those calls.
    vk::SwapchainLatencyCreateInfoNV latency_create_info{VK_TRUE};
    vk::PresentIdKHR present_id_info{1, &present_id};

    std::chrono::duration<double> sleep_time = std::chrono::nanoseconds(0);

    std::vector<vk::LatencyTimingsFrameReportNV> timings;
};

} // namespace merian
