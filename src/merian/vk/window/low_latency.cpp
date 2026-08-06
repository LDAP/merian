#include "merian/vk/window/low_latency.hpp"

#include "merian/utils/chrono.hpp"
#include "merian/utils/stopwatch.hpp"

namespace merian {

namespace {

uint32_t interval_us_for_fps(const uint32_t max_fps) {
    return max_fps == 0 ? 0 : 1000000 / max_fps;
}

} // namespace

LowLatency::LowLatency(const ContextHandle& context)
    : context(context), nv_reflex_supported(context->get_device()->extension_enabled(
                            VK_NV_LOW_LATENCY_2_EXTENSION_NAME)),
      amd_anti_lag_supported(
          context->get_device()->extension_enabled(VK_AMD_ANTI_LAG_EXTENSION_NAME)) {
    if (nv_reflex_supported) {
        sleep_semaphore = TimelineSemaphore::create(context);
    }
}

LowLatency::~LowLatency() {
    mode = LowLatencyMode::OFF;
    update_backend();
}

void LowLatency::set_swapchain(const SwapchainHandle& swapchain) {
    this->swapchain = swapchain;

    if (!swapchain) {
        return;
    }

    SwapchainHooks hooks;
    if (nv_reflex_supported) {
        // Unconditional: the swapchain must be able to enter low latency mode without a recreate
        // when the mode is switched at runtime.
        hooks.pnext_create_info = [this](void* p_next) {
            latency_create_info.pNext = p_next;
            return &latency_create_info;
        };
        hooks.pnext_present_info = [this](void* p_next) -> void* {
            if (applied_backend != Backend::NV_REFLEX) {
                return p_next;
            }
            present_id_info.pNext = p_next;
            return &present_id_info;
        };
    }
    hooks.on_present_begin = [this] { on_present_begin(); };
    hooks.on_present_end = [this] { on_present_end(); };
    swapchain->set_hooks(hooks);
}

LowLatency::Backend LowLatency::resolve_backend() const {
    if (swapchain.expired()) {
        return Backend::NONE;
    }

    switch (mode) {
    case LowLatencyMode::OFF:
        return Backend::NONE;
    case LowLatencyMode::AUTO:
        if (nv_reflex_supported) {
            return Backend::NV_REFLEX;
        }
        return amd_anti_lag_supported ? Backend::AMD_ANTI_LAG : Backend::NONE;
    case LowLatencyMode::NV_REFLEX:
        return nv_reflex_supported ? Backend::NV_REFLEX : Backend::NONE;
    case LowLatencyMode::AMD_ANTI_LAG:
        return amd_anti_lag_supported ? Backend::AMD_ANTI_LAG : Backend::NONE;
    }

    return Backend::NONE;
}

vk::SwapchainKHR LowLatency::update_backend() {
    const Backend backend = resolve_backend();
    const SwapchainHandle swapchain = this->swapchain.lock();
    const vk::SwapchainKHR vk_swapchain =
        swapchain ? swapchain->get_swapchain() : vk::SwapchainKHR{VK_NULL_HANDLE};

    const bool swapchain_changed = applied_swapchain != vk_swapchain;

    if (applied_backend == Backend::AMD_ANTI_LAG && backend != Backend::AMD_ANTI_LAG) {
        context->get_device()->get_device().antiLagUpdateAMD(
            vk::AntiLagDataAMD{vk::AntiLagModeAMD::eOff});
    }
    // A swapchain that is gone does not need to be reset, and its handle may already be destroyed.
    if (applied_backend == Backend::NV_REFLEX && backend != Backend::NV_REFLEX &&
        !swapchain_changed && applied_swapchain) {
        context->get_device()->get_device().setLatencySleepModeNV(
            applied_swapchain, vk::LatencySleepModeInfoNV{VK_FALSE, VK_FALSE, 0});
    }

    if (backend == Backend::NV_REFLEX && vk_swapchain &&
        (applied_backend != backend || swapchain_changed || applied_boost != boost ||
         applied_max_fps != max_fps)) {
        context->get_device()->get_device().setLatencySleepModeNV(
            vk_swapchain, vk::LatencySleepModeInfoNV{VK_TRUE, static_cast<vk::Bool32>(boost),
                                                     interval_us_for_fps(max_fps)});
    }

    applied_backend = backend;
    applied_swapchain = vk_swapchain;
    applied_boost = boost;
    applied_max_fps = max_fps;

    return backend == Backend::NV_REFLEX ? vk_swapchain : vk::SwapchainKHR{VK_NULL_HANDLE};
}

void LowLatency::set_marker(const vk::SwapchainKHR& swapchain, const vk::LatencyMarkerNV marker) {
    if (!swapchain) {
        return;
    }
    context->get_device()->get_device().setLatencyMarkerNV(
        swapchain, vk::SetLatencyMarkerInfoNV{present_id, marker});
}

void LowLatency::anti_lag_update(const vk::AntiLagStageAMD stage) {
    const vk::AntiLagPresentationInfoAMD presentation_info{stage, present_id};
    context->get_device()->get_device().antiLagUpdateAMD(
        vk::AntiLagDataAMD{vk::AntiLagModeAMD::eOn, max_fps, &presentation_info});
}

void LowLatency::begin_frame() {
    present_id++;

    const vk::SwapchainKHR nv_swapchain = update_backend();

    const Stopwatch sw_sleep;
    if (nv_swapchain) {
        sleep_value++;
        context->get_device()->get_device().latencySleepNV(
            nv_swapchain, vk::LatencySleepInfoNV{**sleep_semaphore, sleep_value});
        sleep_semaphore->wait(sleep_value);
    } else if (applied_backend == Backend::AMD_ANTI_LAG) {
        anti_lag_update(vk::AntiLagStageAMD::eInput);
    }
    sleep_time = sleep_time * 0.9 + sw_sleep.duration() * 0.1;

    set_marker(nv_swapchain, vk::LatencyMarkerNV::eSimulationStart);
    set_marker(nv_swapchain, vk::LatencyMarkerNV::eInputSample);
}

void LowLatency::begin_render() {
    const vk::SwapchainKHR nv_swapchain = applied_backend == Backend::NV_REFLEX
                                              ? applied_swapchain
                                              : vk::SwapchainKHR{VK_NULL_HANDLE};
    set_marker(nv_swapchain, vk::LatencyMarkerNV::eSimulationEnd);
    set_marker(nv_swapchain, vk::LatencyMarkerNV::eRendersubmitStart);
}

void LowLatency::on_present_begin() {
    if (applied_backend == Backend::NV_REFLEX) {
        set_marker(applied_swapchain, vk::LatencyMarkerNV::eRendersubmitEnd);
        set_marker(applied_swapchain, vk::LatencyMarkerNV::ePresentStart);
    } else if (applied_backend == Backend::AMD_ANTI_LAG) {
        anti_lag_update(vk::AntiLagStageAMD::ePresent);
    }
}

void LowLatency::on_present_end() {
    if (applied_backend == Backend::NV_REFLEX) {
        set_marker(applied_swapchain, vk::LatencyMarkerNV::ePresentEnd);
    }
}

void LowLatency::properties(Properties& props) {
    props.config_enum("mode", mode, Properties::OptionsStyle::COMBO,
                      "Delays the frame start to reduce input-to-photon latency. 'auto' picks the "
                      "backend the device supports.");

    props.config_bool("boost", boost,
                      "NV Reflex only: keep the GPU at high clocks even when it is not saturated.");
    props.config_uint("max fps", max_fps, "Frame rate the backend paces towards. 0 is uncapped.", 0,
                      1000);

    if (!props.is_ui()) {
        return;
    }

    const auto availability = [](const bool supported) { return supported ? "yes" : "no"; };
    props.output_text("{}: {}\n{}: {}", VK_NV_LOW_LATENCY_2_EXTENSION_NAME,
                      availability(nv_reflex_supported), VK_AMD_ANTI_LAG_EXTENSION_NAME,
                      availability(amd_anti_lag_supported));

    const Backend backend = resolve_backend();
    if (backend == Backend::NONE) {
        if (mode != LowLatencyMode::OFF) {
            props.output_text("inactive: {}", swapchain.expired()
                                                  ? "no swapchain to pace against"
                                                  : "the selected backend is not available");
        }
        return;
    }

    props.output_text("sleep time: {:04f}ms", to_milliseconds(sleep_time));

    if (backend != Backend::NV_REFLEX) {
        return;
    }

    // Reported with a few frames of delay, so the newest complete report is the last entry.
    vk::GetLatencyMarkerInfoNV marker_info{};
    context->get_device()->get_device().getLatencyTimingsNV(applied_swapchain, &marker_info);
    timings.assign(marker_info.timingCount, {});
    if (timings.empty()) {
        return;
    }
    marker_info.pTimings = timings.data();
    context->get_device()->get_device().getLatencyTimingsNV(applied_swapchain, &marker_info);

    const vk::LatencyTimingsFrameReportNV& report = timings.back();
    const auto span_ms = [](const uint64_t begin_us, const uint64_t end_us) {
        return end_us > begin_us ? static_cast<double>(end_us - begin_us) / 1000.0 : 0.0;
    };
    props.output_text(
        "frame {}\ntotal: {:04f}ms\nsimulation: {:04f}ms\nrender submit: {:04f}ms\ndriver: "
        "{:04f}ms\nos render queue: {:04f}ms\ngpu render: {:04f}ms",
        report.presentID, span_ms(report.simStartTimeUs, report.presentEndTimeUs),
        span_ms(report.simStartTimeUs, report.simEndTimeUs),
        span_ms(report.renderSubmitStartTimeUs, report.renderSubmitEndTimeUs),
        span_ms(report.driverStartTimeUs, report.driverEndTimeUs),
        span_ms(report.osRenderQueueStartTimeUs, report.osRenderQueueEndTimeUs),
        span_ms(report.gpuRenderStartTimeUs, report.gpuRenderEndTimeUs));
}

} // namespace merian
