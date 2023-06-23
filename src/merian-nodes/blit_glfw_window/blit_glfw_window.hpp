#pragma once

#include "merian-nodes/blit_external/blit_external.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/window/glfw_window.hpp"
#include "merian/vk/window/swapchain.hpp"

namespace merian {

template <BlitNodeMode mode = FIT> class GLFWWindowNode : public BlitExternalNode<mode> {
  public:
    GLFWWindowNode(const SharedContext context,
                   const GLFWWindowHandle window,
                   const SurfaceHandle surface,
                   const std::optional<QueueHandle> wait_queue = std::nullopt)
        : window(window), surface(surface) {
        swapchain = make_shared<merian::Swapchain>(context, surface, wait_queue);
    }

    virtual std::string name() override {
        return "BlitSwapchainNode";
    }

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             GraphRun& run,
                             const uint32_t set_idx,
                             const std::vector<ImageHandle>& image_inputs,
                             const std::vector<BufferHandle>& buffer_inputs,
                             const std::vector<ImageHandle>& image_outputs,
                             const std::vector<BufferHandle>& buffer_outputs) override {

        auto aquire = swapchain->aquire_auto_resize(*window);
        assert(aquire.has_value());
        BlitExternalNode<mode>::set_target(aquire->image, vk::ImageLayout::eUndefined,
                                           vk::ImageLayout::ePresentSrcKHR,
                                           vk::Extent3D(aquire->extent, 1));
        BlitExternalNode<mode>::cmd_process(cmd, run, set_idx, image_inputs, buffer_inputs,
                                            image_outputs, buffer_outputs);
        run.add_wait_semaphore(aquire->wait_semaphore, vk::PipelineStageFlagBits::eTransfer);
        run.add_signal_semaphore(aquire->signal_semaphore);
        run.add_submit_callback([&](const QueueHandle& queue) { swapchain->present(*queue); });
    }

    SwapchainHandle get_swapchain() {
        return swapchain;
    }

  private:
    GLFWWindowHandle window;
    SurfaceHandle surface;
    SwapchainHandle swapchain;
};

} // namespace merian
