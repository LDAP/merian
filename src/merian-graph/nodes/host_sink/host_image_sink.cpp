#include "merian-graph/nodes/host_sink/host_image_sink.hpp"
#include "merian-graph/graph/errors.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <vulkan/vulkan_format_traits.hpp>

namespace merian {

std::vector<InputConnectorDescriptor> HostImageSink::describe_inputs() {
    return {{.name = "src", .connector = con_src, .access = ConnectorAccess::transfer_src}};
}

HostImageSink::NodeStatusFlags
HostImageSink::on_connected(const NodeIOLayout& io_layout,
                            const NodeIO& io,
                            [[maybe_unused]] const NodeConnectionInfo& info,
                            [[maybe_unused]] Submission& submission) {
    const vk::Format source = io_layout[con_src]->get_create_info_or_throw().format;
    const std::string_view component = vk::componentName(source, 0);
    if (component == "D" || component == "S") {
        throw graph_errors::node_error{fmt::format("{} is no color format", vk::to_string(source))};
    }
    const vk::PhysicalDevice& physical_device =
        io.get_context()->get_physical_device()->get_physical_device();
    const vk::Format target = format == vk::Format::eUndefined ? source : format;
    const vk::FormatFeatureFlags src_features =
        physical_device.getFormatProperties(source).optimalTilingFeatures;
    const auto integer_kind = [](const vk::Format integer_format) {
        const std::string_view numeric = vk::componentNumericFormat(integer_format, 0);
        return numeric == "UINT" || numeric == "SINT" ? numeric : std::string_view{};
    };
    const bool can_blit = (src_features & vk::FormatFeatureFlagBits::eBlitSrc) &&
                          (physical_device.getFormatProperties(target).optimalTilingFeatures &
                           vk::FormatFeatureFlagBits::eBlitDst) &&
                          integer_kind(source) == integer_kind(target);
    if (target != source && !can_blit) {
        throw graph_errors::node_error{
            fmt::format("cannot convert {} into {}", vk::to_string(source), vk::to_string(target))};
    }
    filter = src_features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear
                 ? vk::Filter::eLinear
                 : vk::Filter::eNearest;
    src_format = source;
    blittable = can_blit;
    return {};
}

HostImageSink::NodeStatusFlags HostImageSink::properties(Properties& config) {
    const bool format_changed =
        config.config_enum("format", format, Properties::OptionsStyle::COMBO,
                           "Converts the captures into this format, Undefined keeps the input's.");
    config.config_uint("fit", &fit.width,
                       "Scales the captures down to fit into this extent keeping the aspect "
                       "ratio, 0 keeps the input's extent.",
                       2);
    if (fit.width != 0 && fit.height != 0 && src_format != vk::Format::eUndefined && !blittable) {
        config.output_text("WARN: fit ignored, {} cannot be blitted", vk::to_string(src_format));
    }
    capture_properties(config);
    return format_changed ? NodeStatusFlags{NEEDS_RECONNECT} : NodeStatusFlags{};
}

HostCopy HostImageSink::cmd_capture(const CommandBufferHandle& cmd,
                                    const NodeIO& io,
                                    [[maybe_unused]] const NodeProcessInfo& info) {
    const ImageHandle& src = io[con_src];
    const ResourceAllocatorHandle& allocator = io.get_allocator();
    const StagingMemoryManagerHandle staging = allocator->get_staging();
    const vk::Format capture_format = format == vk::Format::eUndefined ? src->get_format() : format;
    const vk::Extent3D capture_extent = blittable ? fitted(src->get_extent()) : src->get_extent();
    if (capture_format == src->get_format() && capture_extent == src->get_extent()) {
        return cmd_copy_to_host(cmd, staging, src);
    }
    if (!conversion_image || conversion_image->get_format() != capture_format ||
        conversion_image->get_extent() != capture_extent) {
        conversion_image = create_host_conversion_image(allocator, capture_format, capture_extent);
    }
    return cmd_copy_to_host(cmd, staging, src, conversion_image, filter);
}

vk::Extent3D HostImageSink::fitted(const vk::Extent3D& extent) const {
    if (fit.width == 0 || fit.height == 0) {
        return extent;
    }
    const double scale =
        std::min({1.0, static_cast<double>(fit.width) / static_cast<double>(extent.width),
                  static_cast<double>(fit.height) / static_cast<double>(extent.height)});
    return {std::max<uint32_t>(
                1, static_cast<uint32_t>(std::lround(static_cast<double>(extent.width) * scale))),
            std::max<uint32_t>(
                1, static_cast<uint32_t>(std::lround(static_cast<double>(extent.height) * scale))),
            extent.depth};
}

} // namespace merian
