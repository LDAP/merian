#pragma once

#include "vk/extension/extension.hpp"

#include <optional>

class ExtensionStopwatch : public Extension {
  public:
    ExtensionStopwatch() {}
    ~ExtensionStopwatch() {}
    std::string name() const override {
        return "ExtensionStopwatch";
    }
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
    std::optional<uint32_t> get_nanos(vk::Device& device);
    /* Returns the result or none if getQueryPoolResults fails */
    std::optional<double> get_millis(vk::Device& device);
    /* Returns the result or none if getQueryPoolResults fails */
    std::optional<double> get_seconds(vk::Device& device);

  private:
    vk::QueryPool query_pool;
    float timestamp_period;
};
