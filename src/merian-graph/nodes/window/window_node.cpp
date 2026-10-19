#include "merian-graph/nodes/window/window_node.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian/vk/extension/extension_registry.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <algorithm>

namespace merian {

namespace {

constexpr std::string_view BACKEND_AUTO = "auto";

std::string surface_format_name(const vk::SurfaceFormatKHR& format) {
    return fmt::format("{}, {}", vk::to_string(format.format), vk::to_string(format.colorSpace));
}

void select_by_name(const SwapchainHandle& swapchain,
                    const std::string& surface_format,
                    const std::string& present_mode) {
    for (const vk::SurfaceFormatKHR& format : swapchain->get_supported_surface_formats()) {
        if (surface_format_name(format) == surface_format) {
            swapchain->set_new_surface_format(format);
        }
    }
    for (const vk::PresentModeKHR mode : swapchain->get_supported_present_modes()) {
        if (vk::to_string(mode) == present_mode) {
            swapchain->set_new_present_mode(mode);
        }
    }
}

bool config_name(Properties& config,
                 const std::string& id,
                 std::string& name,
                 const std::vector<std::string>& names) {
    if (!config.is_ui()) {
        return config.config_text(id, name);
    }
    const auto it = std::ranges::find(names, name);
    int selected = it == names.end() ? 0 : static_cast<int>(std::distance(names.begin(), it));
    if (!config.config_options(id, selected, names)) {
        return false;
    }
    name = names[selected];
    return true;
}

} // namespace

void WindowNode::initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& /*allocator*/) {
    providers = context->find_providers<WindowProvider>();
    if (providers.empty()) {
        throw graph_errors::node_error{"WindowNode requires at least one WindowProvider"};
    }
}

WindowNode::NodeStatusFlags WindowNode::on_connected(const NodeIOLayout& /*io_layout*/,
                                                     const NodeIO& io,
                                                     const NodeConnectionInfo& info,
                                                     Submission& /*submission*/) {
    const auto& selected = get_selected_provider();
    if (!window || selected != active_provider) {
        const ContextHandle& context = io.get_context();
        const WindowCreateInfo create_info{.title = fmt::format("merian: {}", info.identifier)};
        window = selected->create_window(context->get_device(), create_info);
        active_provider = selected;
        if (fullscreen) {
            window->set_fullscreen(true);
        }

        swapchain_manager.emplace(window->create_swapchain(context));
        select_by_name(get_swapchain(), surface_format, present_mode);
    }
    return {};
}

std::vector<InputConnectorDescriptor> WindowNode::describe_inputs() {
    return {{std::string{DISPLAY_INPUT}, image_in, ConnectorAccess::transfer_src, 0, true}};
}

std::vector<OutputConnectorDescriptor> WindowNode::describe_outputs(const NodeIOLayout& /*io*/) {
    return {{"canvas", con_canvas, ConnectorAccess::transfer_dst},
            {"controller", con_controller},
            {"window", con_window}};
}

WindowNode::NodeStatusFlags WindowNode::pre_process(const NodeIO& io,
                                                    [[maybe_unused]] const NodeProcessInfo& info) {
    if (!window) {
        return {};
    }
    window->poll_events();
    if (!window->should_close()) {
        return {};
    }
    if (on_close_send_event) {
        io.send_event("close");
    }
    return on_close_remove_node ? NodeStatusFlagBits::REMOVE_NODE : NodeStatusFlags{};
}

[[nodiscard]] WindowNode::NodeStatusFlags
WindowNode::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    assert(swapchain_manager);
    NodeStatusFlags flags{};

    const int64_t signed_max_img_count = get_swapchain()->get_max_image_count();
    const int64_t signed_iteration = static_cast<int64_t>(info.get_total_iteration());
    const auto presentable = static_cast<uint64_t>(
        std::max(static_cast<int64_t>(0), signed_iteration - signed_max_img_count + 1));
    throttle = info.get_iteration_semaphore()->get_counter_value() < presentable;
    info.wait_iteration_semaphore(presentable);

    get_swapchain()->set_min_images(info.get_iterations_in_flight());
    std::optional<SwapchainAcquireResult> acquire;
    {
        MERIAN_PROFILE_SCOPE(info.get_profiler(), "acquire");
        acquire = swapchain_manager->acquire(window, acquire_timeout_ns);
    }

    // A minimised window, a timed out acquire or an exhausted recreate leave no image.
    io[con_canvas].set(acquire ? acquire->image_view : nullptr);
    io[con_controller] = window->get_input_controller();
    io[con_window] = window;

    if (acquire) {
        const CommandBufferHandle& cmd = submission.get_cmd();
        const ImageHandle image = acquire->image_view->get_image();

        // Provide the swapchain image as a canvas: blit the optional "src" input, or clear.
        cmd->barrier(image->barrier2(vk::ImageLayout::eTransferDstOptimal, true));

        ImageHandle src_image;
        if (io.is_connected(image_in)) {
            current_src_array_size = io[image_in].get_array_size();
            if (current_src_array_size > 0) {
                src_array_element = std::min(src_array_element, current_src_array_size - 1);
                src_image = io[image_in].get_image(src_array_element);
            }
        } else {
            current_src_array_size = 0;
        }

        {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "blit/clear");
            if (src_image) {
                const vk::Filter filter =
                    src_image->format_features() &
                            vk::FormatFeatureFlagBits::eSampledImageFilterLinear
                        ? vk::Filter::eLinear
                        : vk::Filter::eNearest;
                cmd_blit(mode, cmd, src_image, vk::ImageLayout::eGeneral, src_image->get_extent(),
                         image, vk::ImageLayout::eTransferDstOptimal, image->get_extent(),
                         vk::ClearColorValue{}, filter);
            } else {
                cmd->clear(image);
            }
        }

        // Transition to present once every node wrote the canvas.
        submission.add_pre_submit_callback(
            [image,
             present_layout = get_swapchain()->get_present_layout()](Submission& submission) {
                submission.get_cmd()->barrier(image->barrier2(present_layout));
            });

        submission.add_wait_semaphore(acquire->wait_semaphore,
                                      vk::PipelineStageFlagBits::eTransfer);
        submission.add_signal_semaphore(acquire->signal_semaphore);

        const uint32_t index = acquire->index;
        const SwapchainHandle swapchain = get_swapchain();
        submission.add_submit_callback(
            [index, swapchain, &info](const QueueHandle& queue, Submission& /*submission*/) {
                try {
                    MERIAN_PROFILE_SCOPE(info.get_profiler(), "present");
                    swapchain->present(queue, index);
                } catch (const Swapchain::needs_recreate&) {
                    return;
                }
            });

        if (request_rebuild_on_recreate && acquire->did_recreate)
            flags |= NodeStatusFlagBits::NEEDS_RECONNECT;
    }
    return flags;
}

const SwapchainHandle& WindowNode::get_swapchain() {
    assert(swapchain_manager);
    return swapchain_manager->get_swapchain();
}

WindowNode::NodeStatusFlags WindowNode::properties(Properties& config) {
    if (throttle) {
        config.output_text("WARN: throttling CPU, too many frames in flight for swapchain!");
    }

    NodeStatusFlags flags{};

    if (current_src_array_size > 0 || !config.is_ui()) {
        config.config_uint("source array element", src_array_element, "", 0u,
                           std::max(current_src_array_size, 1u) - 1);
    }

    if (window) {
        fullscreen = window->is_fullscreen();
    }
    if (window || !config.is_ui()) {
        int selected_mode = static_cast<int>(fullscreen);
        if (config.config_options("mode", selected_mode, {"windowed", "fullscreen"})) {
            fullscreen = selected_mode != 0;
            if (window) {
                window->set_fullscreen(fullscreen);
            }
        }
    }

    int int_mode = mode;
    config.config_options("blit mode", int_mode, {"FIT", "FILL", "STRETCH"},
                          Properties::OptionsStyle::LIST_BOX);
    mode = static_cast<BlitMode>(int_mode);

    if (swapchain_manager || !config.is_ui()) {
        std::vector<std::string> surface_formats;
        std::vector<std::string> present_modes;
        if (swapchain_manager) {
            const SwapchainHandle& swapchain = get_swapchain();
            surface_format = surface_format_name(swapchain->get_new_surface_format());
            present_mode = vk::to_string(swapchain->get_new_present_mode());
            for (const vk::SurfaceFormatKHR& format : swapchain->get_supported_surface_formats()) {
                surface_formats.emplace_back(surface_format_name(format));
            }
            for (const vk::PresentModeKHR supported_mode :
                 swapchain->get_supported_present_modes()) {
                present_modes.emplace_back(vk::to_string(supported_mode));
            }
        }
        const bool format_changed =
            config_name(config, "surface format", surface_format, surface_formats);
        const bool present_mode_changed =
            config_name(config, "present mode", present_mode, present_modes);
        if (swapchain_manager && (format_changed || present_mode_changed)) {
            select_by_name(get_swapchain(), surface_format, present_mode);
        }
    }

    config.config_bool("rebuild on recreate", request_rebuild_on_recreate,
                       "requests a graph rebuild if the swapchain was recreated.");

    config.config_uint64("acquire timeout", acquire_timeout_ns, "in nanoseconds");

    if (config.st_begin_child("on close", "On close")) {
        config.config_bool("send event", on_close_send_event,
                           "Sends \"close\" each iteration while the window should close.");
        config.config_bool("remove node", on_close_remove_node);
        config.st_end_child();
    }

    if (swapchain_manager) {
        const std::optional<Swapchain::SwapchainInfo>& swapchain_info =
            get_swapchain()->get_swapchain_info();
        if (swapchain_info) {
            config.output_text(fmt::format(
                "surface format: {}\ncolor space: {}\nimage count: "
                "{}\nextent: {}x{}\npresent mode: {}",
                vk::to_string(swapchain_info->surface_format.format),
                vk::to_string(swapchain_info->surface_format.colorSpace),
                swapchain_info->images.size(), swapchain_info->extent.width,
                swapchain_info->extent.height, vk::to_string(swapchain_info->present_mode)));
        }
    }

    const auto& registry = ExtensionRegistry::get_instance();
    std::vector<std::string> backends = {std::string{BACKEND_AUTO}};
    for (const auto& provider : providers) {
        backends.emplace_back(registry.get_name(provider));
    }
    if (config.config_options("backend", selected_backend, backends,
                              Properties::OptionsStyle::LIST_BOX,
                              "auto: the window provider of the highest priority")) {
        flags |= NodeStatusFlagBits::NEEDS_RECONNECT;
    }
    if (active_provider) {
        config.output_text("active backend: {}", registry.get_name(active_provider));
    }

    return flags;
}

const std::shared_ptr<WindowProvider>& WindowNode::get_selected_provider() const {
    const int provider = std::max(selected_backend - 1, 0);
    return providers[std::min(provider, static_cast<int>(providers.size()) - 1)];
}

} // namespace merian
