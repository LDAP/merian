#pragma once

#include "merian-nodes/connectors/image/vk_image_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/utils/blits.hpp"
#include "merian/vk/window/glfw_window.hpp"
#include "merian/vk/window/swapchain.hpp"
#include "merian/vk/window/swapchain_manager.hpp"
#include <csignal>

namespace merian {

/*
 * Outputs to a GLFW window.
 * This node requires the error handling features of ExtensionVkGLFW
 */
class GLFWWindowNode : public Node {
  public:
    GLFWWindowNode(const ContextHandle& context) : Node() {
        const auto glfw_ext = context->get_extension<ExtensionGLFW>();
        if (glfw_ext) {
            window = glfw_ext->create_window();

            const SwapchainHandle swapchain =
                std::make_shared<merian::Swapchain>(context, window->get_surface());
            swapchain_manager.emplace(swapchain);
        }
    }

    virtual std::vector<InputConnectorHandle> describe_inputs() override {
        if (!window) {
            throw graph_errors::node_error{"node requires ExtensionVkGLFW context extension"};
        }

        return {image_in};
    }

    virtual NodeStatusFlags pre_process([[maybe_unused]] const GraphRun& run,
                                        [[maybe_unused]] const NodeIO& io) override {

        if (on_should_close_remove_node && window && window->should_close()) {
            return NodeStatusFlagBits::REMOVE_NODE;
        }

        return {};
    }

    virtual void process(GraphRun& run,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) override {
        assert(swapchain_manager);

        const std::optional<SwapchainAcquireResult> acquire =
            swapchain_manager->acquire(window, acquire_timeout_ns);

        if (acquire) {
            const CommandBufferHandle& cmd = run.get_cmd();
            const ImageHandle image = acquire->image_view->get_image();

            auto barrier = image->barrier2(vk::ImageLayout::eTransferDstOptimal, true);
            cmd->barrier(barrier);

            ImageHandle src_image;
            if (io.is_connected(image_in)) {
                current_src_array_size = io[image_in].get_array_size();
                src_array_element = std::min(src_array_element, current_src_array_size - 1);
                src_image = io[image_in].get_image(src_array_element);
            } else {
                current_src_array_size = 0;
            }

            if (src_image) {
                const vk::Filter filter =
                    src_image->format_features() &
                            vk::FormatFeatureFlagBits::eSampledImageFilterLinear
                        ? vk::Filter::eLinear
                        : vk::Filter::eNearest;

                cmd_blit(mode, cmd, src_image, vk::ImageLayout::eTransferSrcOptimal,
                         src_image->get_extent(), image, vk::ImageLayout::eTransferDstOptimal,
                         image->get_extent(), vk::ClearColorValue{}, filter);
            } else {
                cmd->clear(image);
            }

            cmd->barrier(image->barrier2(vk::ImageLayout::ePresentSrcKHR));

            on_blit_completed(cmd, *acquire);

            run.add_wait_semaphore(acquire->wait_semaphore, vk::PipelineStageFlagBits::eTransfer);
            run.add_signal_semaphore(acquire->signal_semaphore);

            uint32_t index = acquire->index;
            SwapchainHandle swapchain = get_swapchain();
            run.add_submit_callback([index, swapchain](const QueueHandle& queue, GraphRun& run) {
                try {
                    Stopwatch present_duration;
                    swapchain->present(queue, index);
                    run.hint_external_wait_time(present_duration.duration());
                } catch (const Swapchain::needs_recreate& e) {
                    // do nothing and hope for the best
                    return;
                }
            });

            if (request_rebuild_on_recreate && acquire->did_recreate)
                run.request_reconnect();
        }

        if (window && window->should_close()) {
            if (on_should_close_sigint) {
                raise(SIGINT);
            }
            if (on_should_close_sigterm) {
                raise(SIGTERM);
            }
        }
    }

    const SwapchainHandle& get_swapchain() {
        assert(swapchain_manager && "ExtensionVkGLFW not avaliable");
        return swapchain_manager->get_swapchain();
    }

    NodeStatusFlags properties(Properties& config) override {
        if (current_src_array_size > 0) {
            config.config_uint("source array element", src_array_element, 0,
                               current_src_array_size - 1);
        }

        GLFWmonitor* monitor = window ? glfwGetWindowMonitor(*window) : nullptr;
        int fullscreen = static_cast<int>(monitor != nullptr);
        const int old_fullscreen = fullscreen;
        config.config_options("mode", fullscreen, {"windowed", "fullscreen"});
        if (window && fullscreen != old_fullscreen) {
            if (fullscreen != 0) {
                try {
                    glfwGetWindowPos(*window, &windowed_pos_size[0], &windowed_pos_size[1]);
                } catch (const ExtensionGLFW::glfw_error& e) {
                    if (e.id != GLFW_FEATURE_UNAVAILABLE) {
                        throw e;
                    }
                    windowed_pos_size[0] = windowed_pos_size[1] = 0;
                }
                glfwGetWindowSize(*window, &windowed_pos_size[2], &windowed_pos_size[3]);
                monitor = glfwGetPrimaryMonitor();
                const GLFWvidmode* vidmode = glfwGetVideoMode(monitor);
                glfwSetWindowMonitor(*window, monitor, 0, 0, vidmode->width, vidmode->height,
                                     vidmode->refreshRate);
            } else {
                glfwSetWindowMonitor(*window, NULL, windowed_pos_size[0], windowed_pos_size[1],
                                     windowed_pos_size[2], windowed_pos_size[3], GLFW_DONT_CARE);
            }
        }

        int int_mode = mode;
        config.config_options("blit mode", int_mode, {"FIT", "FILL", "STRETCH"},
                              Properties::OptionsStyle::LIST_BOX);
        mode = (BlitMode)int_mode;

        if (swapchain_manager) {
            const SwapchainHandle swapchain = get_swapchain();

            int selected;
            std::vector<std::string> str_surface_formats;
            const auto& surface_formats = swapchain->get_supported_surface_formats();
            for (uint32_t i = 0; i < surface_formats.size(); i++) {
                const vk::SurfaceFormatKHR& format = surface_formats[i];
                str_surface_formats.emplace_back(fmt::format("{}, {}", vk::to_string(format.format),
                                                             vk::to_string(format.colorSpace)));
                if (format == swapchain->get_new_surface_format()) {
                    selected = (int)i;
                }
            }
            if (config.config_options("surface format", selected, str_surface_formats)) {
                swapchain->set_new_surface_format(surface_formats[selected]);
            }

            std::vector<std::string> str_present_modes;
            const auto& present_modes = swapchain->get_supported_present_modes();
            for (uint32_t i = 0; i < present_modes.size(); i++) {
                const vk::PresentModeKHR& mode = present_modes[i];
                str_present_modes.emplace_back(fmt::format("{}", vk::to_string(mode)));
                if (mode == swapchain->get_new_present_mode()) {
                    selected = (int)i;
                }
            }
            if (config.config_options("present mode", selected, str_present_modes)) {
                swapchain->set_new_present_mode(present_modes[selected]);
            }
        }

        config.config_bool("rebuild on recreate", request_rebuild_on_recreate,
                           "requests a graph rebuild if the swapchain was recreated.");

        config.config_uint64("acquire timeout", acquire_timeout_ns, "in nanoseconds");

        if (config.st_begin_child("on_should_close_actions", "On should_close()")) {
            config.config_bool("send sigint", on_should_close_sigint);
            config.config_bool("send sigterm", on_should_close_sigterm);
            config.config_bool("remove node", on_should_close_remove_node);

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

        return {};
    }

    // Window can be nullptr if GLFW extension is not available
    const GLFWWindowHandle& get_window() const {
        return window;
    }

    // Set a callback for when the blit of the node input was completed.
    // The image will have vk::ImageLayout::ePresentSrcKHR.
    void
    set_on_blit_completed(const std::function<void(const CommandBufferHandle& cmd,
                                                   const SwapchainAcquireResult& acquire_result)>&
                              on_blit_completed) {
        this->on_blit_completed = on_blit_completed;
    }

  private:
    uint32_t src_array_element = 0;
    uint32_t current_src_array_size = 1;

    GLFWWindowHandle window = nullptr;
    std::optional<SwapchainManager> swapchain_manager = std::nullopt;

    BlitMode mode = FIT;

    std::function<void(const CommandBufferHandle& cmd,
                       const SwapchainAcquireResult& acquire_result)>
        on_blit_completed = []([[maybe_unused]] const CommandBufferHandle& cmd,
                               [[maybe_unused]] const SwapchainAcquireResult& acquire_result) {};

    VkImageInHandle image_in = VkImageIn::transfer_src("src", 0, true);

    std::array<int, 4> windowed_pos_size;
    bool request_rebuild_on_recreate = false;
    uint64_t acquire_timeout_ns = 1000L * 1000L * 100L; // .1s

    bool on_should_close_sigint = false;
    bool on_should_close_sigterm = false;
    bool on_should_close_remove_node = true;
};

} // namespace merian
