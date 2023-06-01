#pragma once

#include "merian/vk/extension/extension.hpp"

#include <optional>

namespace merian {

class ExtensionStopwatch : public Extension {
  public:
    ExtensionStopwatch(uint32_t number_stopwatches = 1)
        : Extension("ExtensionStopwatch"), number_stopwatches(number_stopwatches) {}
    ~ExtensionStopwatch() {}
    void on_context_created(const Context&) override;
    void on_destroy_context(const Context&) override;

  public: // own methods
    /* Resets the query pool and writes the first timestamp */
    void start_stopwatch(vk::CommandBuffer& cb,
                         vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands, uint32_t stopwatch_id = 0);
    /* Writes the second timestamp */
    void stop_stopwatch(vk::CommandBuffer& cb,
                        vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands, uint32_t stopwatch_id = 0);
    /* Returns the result or none if getQueryPoolResults fails */
    uint32_t get_nanos(uint32_t stopwatch_id = 0);
    /* Returns the result or none if getQueryPoolResults fails */
    double get_millis(uint32_t stopwatch_id = 0);
    /* Returns the result or none if getQueryPoolResults fails */
    double get_seconds(uint32_t stopwatch_id = 0);

  private:
    uint32_t number_stopwatches;
    vk::QueryPool query_pool;
    float timestamp_period;
    vk::Device device = VK_NULL_HANDLE;
};

} // namespace merian
