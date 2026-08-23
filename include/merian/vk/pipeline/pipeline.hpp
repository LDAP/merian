#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include <spdlog/spdlog.h>

#include <cstdlib>

namespace merian {

// MERIAN_SHADER_STATS=1 makes every pipeline report the driver's compilation statistics (register
// counts, occupancy, spilling) once at creation.
inline bool shader_stats_requested() {
    static const bool requested = std::getenv("MERIAN_SHADER_STATS") != nullptr;
    return requested;
}

class Pipeline : public std::enable_shared_from_this<Pipeline>, public Object {

  public:
    Pipeline(const ContextHandle& context,
             const std::shared_ptr<PipelineLayout>& pipeline_layout,
             const vk::PipelineCreateFlags flags)
        : context(context), pipeline_layout(pipeline_layout), flags(flags) {}

    virtual ~Pipeline() {};

    // ---------------------------------------------------------------------------

    operator const vk::Pipeline&() const {
        return pipeline;
    }

    const vk::Pipeline& get_pipeline() const {
        return pipeline;
    }

    const std::shared_ptr<PipelineLayout>& get_layout() const {
        return pipeline_layout;
    }

    bool supports_descriptor_buffer() const {
        return bool(flags & vk::PipelineCreateFlagBits::eDescriptorBufferEXT);
    }

    bool supports_descriptor_set() const {
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreateFlagBits.html#
        return !supports_descriptor_buffer();
    }

    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const = 0;

    virtual vk::PipelineStageFlags get_pipeline_stage_flags() const = 0;

    virtual vk::PipelineStageFlags2 get_pipeline_stage_flags2() const = 0;

  protected:
    // Adds the flag the statistics query needs; the driver compiles the same code either way.
    static vk::PipelineCreateFlags capture_statistics(const ContextHandle& context,
                                                      const vk::PipelineCreateFlags flags) {
        if (shader_stats_requested() &&
            context->get_device()->get_enabled_features().get_feature("pipelineExecutableInfo")) {
            return flags | vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
        }
        return flags;
    }

    static std::string format_statistic(const vk::PipelineExecutableStatisticKHR& stat) {
        switch (stat.format) {
        case vk::PipelineExecutableStatisticFormatKHR::eBool32:
            return stat.value.b32 ? "true" : "false";
        case vk::PipelineExecutableStatisticFormatKHR::eInt64:
            return fmt::format("{}", stat.value.i64);
        case vk::PipelineExecutableStatisticFormatKHR::eUint64:
            return fmt::format("{}", stat.value.u64);
        case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
            return fmt::format("{}", stat.value.f64);
        }
        return "?";
    }

    void log_statistics(const std::string& label) const {
        if (!bool(flags & vk::PipelineCreateFlagBits::eCaptureStatisticsKHR)) {
            return;
        }
        const vk::Device device = context->get_device()->get_device();
        const auto executables =
            device.getPipelineExecutablePropertiesKHR(vk::PipelineInfoKHR{pipeline});
        for (uint32_t index = 0; index < executables.size(); index++) {
            std::string stats;
            for (const auto& stat : device.getPipelineExecutableStatisticsKHR(
                     vk::PipelineExecutableInfoKHR{pipeline, index})) {
                stats += fmt::format("\n    {}: {}", stat.name.data(), format_statistic(stat));
            }
            SPDLOG_INFO("shader stats {} [{}]{}", label, executables[index].name.data(), stats);
        }
    }

    const ContextHandle context;
    const PipelineLayoutHandle pipeline_layout;
    const vk::PipelineCreateFlags flags;

    vk::Pipeline pipeline;
};

using PipelineHandle = std::shared_ptr<Pipeline>;

} // namespace merian
