#pragma once

#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/ptr_out.hpp"
#include "merian-graph/graph/node.hpp"

#include "merian/utils/stopwatch.hpp"
#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/imgui/imgui_merian_backend.hpp"
#include "merian/vk/imgui/imgui_renderer.hpp"
#include "merian/vk/window/swapchain_manager.hpp"
#include "merian/vk/window/window.hpp"

namespace merian {

// Draws a UI window onto a Window node's acquired swapchain image. Each frame it sends a graph
// event carrying a Properties; listeners (the graph, plugins) render their controls into it.
class ImGuiNode : public Node {
  public:
    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    ImGuiContextHandle imgui_ctx;
    std::shared_ptr<ImGuiRenderer> imgui_renderer;
    std::shared_ptr<ImGuiMerianBackend> imgui_backend;
    WindowHandle current_window;
    Stopwatch frametime;

    std::string imgui_event = "ui";

    PtrInHandle<SwapchainAcquireResult> con_acquire = PtrIn<SwapchainAcquireResult>::create();
    PtrInHandle<Window> con_window = PtrIn<Window>::create();
    PtrOutHandle<SwapchainAcquireResult> con_acquire_out = PtrOut<SwapchainAcquireResult>::create();
};

} // namespace merian
