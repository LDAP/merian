#include "merian/vk/extension/extension_stopwatch.hpp"
#include "merian/vk/utils/check_result.hpp"

#define SW_QUERY_COUNT 2

namespace merian {

void ExtensionStopwatch::on_context_created(const SharedContext context) {
    this->context = context;

    vk::QueryPoolCreateInfo createInfo({}, vk::QueryType::eTimestamp, SW_QUERY_COUNT);
    query_pool = context->device.createQueryPool(createInfo);

    timestamp_period =
        context->pd_container.physical_device_props.properties.limits.timestampPeriod;
}

void ExtensionStopwatch::on_destroy_context() {
    context->device.destroyQueryPool(query_pool);
    this->context.reset();
}

void ExtensionStopwatch::start_stopwatch(vk::CommandBuffer& cb,
                                         vk::PipelineStageFlagBits pipeline_stage,
                                         uint32_t stopwatch_id) {
    cb.resetQueryPool(query_pool, stopwatch_id * SW_QUERY_COUNT, SW_QUERY_COUNT);
    cb.writeTimestamp(pipeline_stage, query_pool, stopwatch_id * SW_QUERY_COUNT);
}

void ExtensionStopwatch::stop_stopwatch(vk::CommandBuffer& cb,
                                        vk::PipelineStageFlagBits pipeline_stage,
                                        uint32_t stopwatch_id) {
    cb.writeTimestamp(pipeline_stage, query_pool, stopwatch_id * SW_QUERY_COUNT + 1);
}

uint32_t ExtensionStopwatch::get_nanos(uint32_t stopwatch_id) {
    assert(stopwatch_id < number_stopwatches);

    uint64_t timestamps[SW_QUERY_COUNT];
    check_result(context->device.getQueryPoolResults(
                     query_pool, stopwatch_id * SW_QUERY_COUNT, SW_QUERY_COUNT, sizeof(timestamps),
                     &timestamps, sizeof(timestamps[0]), vk::QueryResultFlagBits::e64),
                 "could not get query results");

    uint64_t timediff = timestamps[1] - timestamps[0];
    return timestamp_period * timediff;
}
double ExtensionStopwatch::get_millis(uint32_t stopwatch_id) {
    return get_nanos(stopwatch_id) / (double)1e6;
}

double ExtensionStopwatch::get_seconds(uint32_t stopwatch_id) {
    return get_nanos(stopwatch_id) / (double)1e9;
}

} // namespace merian
