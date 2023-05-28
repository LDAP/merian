#pragma once

#include "vk/extension/extension.hpp"

#include <optional>

namespace merian {

class ExtensionStopwatch : public Extension {
  public:
    ExtensionStopwatch() : Extension("ExtensionStopwatch") {}
    ~ExtensionStopwatch() {}
    void on_context_created(const Context&) override;
    void on_destroy_context(const Context&) override;

  public: // own methods
    /* Resets the query pool and writes the first timestamp */
    void start_stopwatch(vk::CommandBuffer& cb,
                         vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands);
    /* Writes the second timestamp */
    void stop_stopwatch(vk::CommandBuffer& cb,
                        vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands);
    /* Returns the result or none if getQueryPoolResults fails */
    std::optional<uint32_t> get_nanos();
    /* Returns the result or none if getQueryPoolResults fails */
    std::optional<double> get_millis();
    /* Returns the result or none if getQueryPoolResults fails */
    std::optional<double> get_seconds();

  private:
    vk::QueryPool query_pool;
    float timestamp_period;
    vk::Device device = VK_NULL_HANDLE;
};

} // namespace merian
