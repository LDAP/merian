#include "merian/vk/utils/stopwatch_vk.hpp"
#include "merian/vk/utils/check_result.hpp"

#define SW_QUERY_COUNT 2

namespace merian {

StopwatchVk::StopwatchVk(const SharedContext context, uint32_t number_stopwatches)
    : context(context), number_stopwatches(number_stopwatches) {
    vk::QueryPoolCreateInfo createInfo({}, vk::QueryType::eTimestamp,
                                       number_stopwatches * SW_QUERY_COUNT);
    query_pool = context->device.createQueryPool(createInfo);

    timestamp_period =
        context->pd_container.physical_device_props.properties.limits.timestampPeriod;
}

StopwatchVk::~StopwatchVk() {
    context->device.destroyQueryPool(query_pool);
}

void StopwatchVk::start_stopwatch(vk::CommandBuffer& cb,
                                  vk::PipelineStageFlagBits pipeline_stage,
                                  uint32_t stopwatch_id) {
    cb.resetQueryPool(query_pool, stopwatch_id * SW_QUERY_COUNT, SW_QUERY_COUNT);
    cb.writeTimestamp(pipeline_stage, query_pool, stopwatch_id * SW_QUERY_COUNT);
}

void StopwatchVk::stop_stopwatch(vk::CommandBuffer& cb,
                                 vk::PipelineStageFlagBits pipeline_stage,
                                 uint32_t stopwatch_id) {
    cb.writeTimestamp(pipeline_stage, query_pool, stopwatch_id * SW_QUERY_COUNT + 1);
}

uint32_t StopwatchVk::get_nanos(uint32_t stopwatch_id) {
    assert(stopwatch_id < number_stopwatches);

    uint64_t timestamps[SW_QUERY_COUNT];
    check_result(context->device.getQueryPoolResults(
                     query_pool, stopwatch_id * SW_QUERY_COUNT, SW_QUERY_COUNT, sizeof(timestamps),
                     &timestamps, sizeof(timestamps[0]), vk::QueryResultFlagBits::e64),
                 "could not get query results");

    uint64_t timediff = timestamps[1] - timestamps[0];
    return timestamp_period * timediff;
}
double StopwatchVk::get_millis(uint32_t stopwatch_id) {
    return get_nanos(stopwatch_id) / (double)1e6;
}

double StopwatchVk::get_seconds(uint32_t stopwatch_id) {
    return get_nanos(stopwatch_id) / (double)1e9;
}

} // namespace merian
