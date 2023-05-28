#include <vk/extension/extension_stopwatch.hpp>

#define SW_QUERY_COUNT 2

namespace merian {

void ExtensionStopwatch::on_context_created(const Context& context) {
    this->device = context.device;

    vk::QueryPoolCreateInfo createInfo({}, vk::QueryType::eTimestamp, SW_QUERY_COUNT);
    query_pool = context.device.createQueryPool(createInfo);

    vk::PhysicalDeviceProperties properties = context.physical_device.getProperties();
    timestamp_period = properties.limits.timestampPeriod;
}

void ExtensionStopwatch::on_destroy_context(const Context& context) {
    context.device.destroyQueryPool(query_pool);

    this->device = VK_NULL_HANDLE;
}

void ExtensionStopwatch::start_stopwatch(vk::CommandBuffer& cb, vk::PipelineStageFlagBits pipeline_stage) {
    cb.resetQueryPool(query_pool, 0, 2);
    cb.writeTimestamp(pipeline_stage, query_pool, 0);
}

void ExtensionStopwatch::stop_stopwatch(vk::CommandBuffer& cb, vk::PipelineStageFlagBits pipeline_stage) {
    cb.writeTimestamp(pipeline_stage, query_pool, 1);
}

std::optional<uint32_t> ExtensionStopwatch::get_nanos() {
    uint64_t timestamps[2];
    vk::Result result = device.getQueryPoolResults(query_pool, 0, 2, sizeof(timestamps), &timestamps,
                                                   sizeof(timestamps[0]), vk::QueryResultFlagBits::e64);
    if (result != vk::Result::eSuccess)
        return std::nullopt;

    uint64_t timediff = timestamps[1] - timestamps[0];
    return timestamp_period * timediff;
}
std::optional<double> ExtensionStopwatch::get_millis() {
    std::optional<uint32_t> nanos = get_nanos();
    if (nanos.has_value()) {
        return nanos.value() / (double)1e6;
    } else {
        return std::nullopt;
    }
}
std::optional<double> ExtensionStopwatch::get_seconds() {
    std::optional<uint32_t> nanos = get_nanos();
    if (nanos.has_value()) {
        return nanos.value() / (double)1e9;
    } else {
        return std::nullopt;
    }
}

} // namespace merian
