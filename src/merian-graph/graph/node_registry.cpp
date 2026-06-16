#include "merian-graph/graph/node_registry.hpp"

#include "merian-graph/plugin/node_plugin.hpp"
#include "merian/plugin/plugins.hpp"

#include <spdlog/spdlog.h>

#include "merian-graph/nodes/ab_compare/ab_compare.hpp"
#include "merian-graph/nodes/accumulate/accumulate.hpp"
#include "merian-graph/nodes/bloom/bloom.hpp"
#include "merian-graph/nodes/color_image/color_output.hpp"
#include "merian-graph/nodes/exposure/exposure.hpp"
#include "merian-graph/nodes/fbx_scene/fbx_scene_node.hpp"
#include "merian-graph/nodes/fxaa/fxaa.hpp"
#include "merian-graph/nodes/gbuffer_debug/gbuffer_debug.hpp"
#include "merian-graph/nodes/gbuffer_rt/gbuffer.hpp"
#include "merian-graph/nodes/gltf_scene/gltf_scene_node.hpp"
#include "merian-graph/nodes/image_read/hdr_image.hpp"
#include "merian-graph/nodes/image_read/ldr_image.hpp"
#include "merian-graph/nodes/image_write/image_write.hpp"
#include "merian-graph/nodes/imgui/imgui_node.hpp"
#include "merian-graph/nodes/mean/mean.hpp"
#include "merian-graph/nodes/median_approx/median.hpp"
#include "merian-graph/nodes/reduce/reduce.hpp"
#include "merian-graph/nodes/render_pt/render_pt.hpp"
#include "merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.hpp"
#include "merian-graph/nodes/render_restir_di/render_restir_di.hpp"
#include "merian-graph/nodes/shadertoy/shadertoy.hpp"
#include "merian-graph/nodes/svgf/svgf.hpp"
#include "merian-graph/nodes/swapchain_blit/swapchain_blit.hpp"
#include "merian-graph/nodes/taa/taa.hpp"
#include "merian-graph/nodes/tonemap/tonemap.hpp"
#include "merian-graph/nodes/vkdt_filmcurv/vkdt_filmcurv.hpp"
#include "merian-graph/nodes/window/window_node.hpp"

namespace merian {

NodeRegistry& NodeRegistry::get_instance() {
    static NodeRegistry instance;
    return instance;
}

void NodeRegistry::load_from_plugins() {
    static bool loaded = false;
    if (loaded) {
        return;
    }
    loaded = true;

    load_plugins("merian_register_nodes", "merian_node_plugin_abi_version",
                 MERIAN_NODE_PLUGIN_ABI_VERSION, "node", [this](void* register_sym) {
                     reinterpret_cast<merian_register_nodes_fn>(register_sym)(*this);
                 });
}

NodeRegistry::NodeRegistry() {
    register_node_type<ABSplit>("AB Split", "Compare two inputs in a split-view.");
    register_node_type<ABSideBySide>("AB Side By Side",
                                     "Compare two inputs in a side by side view.");
    register_node_type<Accumulate>("Accumulate", "Accumulate values across multiple iterations.");
    register_node_type<Bloom>("Bloom", "Selectively blurs pixels that surpass a threshold.");
    register_node_type<ColorImage>("Color",
                                   "Outputs a image filled cleared with the selected color.");
    register_node_type<AutoExposure>(
        "Exposure", "Exposure with camera-like controls. Includes a robust auto-exposure mode");
    register_node_type<FXAA>("FXAA",
                             "Fast approximate anti-aliasing (FXAA) is a screen-space "
                             "anti-aliasing algorithm created by Timothy Lottes at NVIDIA.");
    register_node_type<GBufferDebugNode>(
        "GBuffer Debug", "Visualize GBuffer fields (normal, depth, hit info, motion vectors).");
    register_node_type<GBufferRTNode>(
        "GBuffer (Raytraced)", "Creates a GBuffer for the Merian scene format using Raytracing.");
    register_node_type<GLTFSceneNode>("GLTF Scene", "Loads a glTF/GLB scene file.");
    register_node_type<FBXSceneNode>("FBX Scene", "Loads an FBX scene file.");
    register_node_type<WindowNode>("Window", "Outputs to a window.");
    register_node_type<HDRImageRead>("HDR Image", "Loads an HDR image.");
    register_node_type<LDRImageRead>("LDR Image", "Loads a LDR image.");
    register_node_type<ImageWrite>("Image Write", "Stores a graph output as image file.");
    register_node_type<ImGuiNode>("ImGui",
                                  "Renders ImGui draw callbacks onto a window's acquired image.");
    register_node_type<MeanToBuffer>(
        "Mean", "Computes the mean of an image and outputs it as a single buffer element.");
    register_node_type<MedianApproxNode>("Median (Approximation)",
                                         "Computes an approximation of the median of a component.");
    register_node_type<Reduce>("Reduce", "Reduce values of multiple input images.");
    register_node_type<RenderMCPG>("Render (Path-traced, MCPG)",
                                   "Path-traced renderer using Markov chain path-guiding.");
    register_node_type<RenderPT>("Render (Path-traced)",
                                 "Path-traced renderer using BSDF-sampling path tracer.");
    register_node_type<RenderRestirDI>(
        "Render (ReSTIR DI)",
        "Screen-space ReSTIR direct illumination with spatiotemporal reservoir reuse.");
    register_node_type<Shadertoy>("Shadertoy",
                                  "Execute Shadertoy-like shaders (Limited implementation).");
    register_node_type<SVGF>("Denoiser (SVGF)", "Spatiotemporal Variance-Guided Filtering.");
    register_node_type<SwapchainBlit>(
        "Swapchain Blit", "Blits a graph image onto a Window node's acquired swapchain image.");
    register_node_type<TAA>("TAA", "Temporal Anti-Aliasing.");
    register_node_type<Tonemap>("Tonemap", "Convert a HDR image to LDR using various tonemaps.");
    register_node_type<VKDTFilmcurv>("Curves", "Adjust brightness and contrast. Ported from VKDT.");

    register_node<Reduce>(
        "Add", "Add values of multiple input images.",
        nlohmann::json({{"initial value", ""}, {"reduction", "accumulator + current_value"}}));
    register_node<Reduce>(
        "Multiply", "Multiply values of multiple input images.",
        nlohmann::json({{"initial value", ""}, {"reduction", "accumulator * current_value"}}));
    register_node<Reduce>(
        "Divide", "Divide values of multiple input images.",
        nlohmann::json({{"initial value", ""}, {"reduction", "accumulator / current_value"}}));
    register_node<Reduce>(
        "Subtract", "Subtract values of multiple input images.",
        nlohmann::json({{"initial value", ""}, {"reduction", "accumulator - current_value"}}));
    register_node<Reduce>(
        "Min", "Compute minimum over all input images.",
        nlohmann::json({{"initial value", ""}, {"reduction", "min(accumulator, current_value)"}}));
    register_node<Reduce>(
        "Max", "Compute maximum over all input images.",
        nlohmann::json({{"initial value", ""}, {"reduction", "max(accumulator, current_value)"}}));

    register_node<Reduce>(
        "Alpha Blend (Premultiplied, Front-to-Back)",
        "Porter-Duff over compositing of multiple images, input 0 on top. Inputs must be "
        "premultiplied.",
        nlohmann::json({{"initial value", "vec4(0)"},
                        {"reduction", "accumulator + current_value * (1.0 - accumulator.a)"}}));
    register_node<Reduce>(
        "Alpha Blend (Premultiplied, Back-to-Front)",
        "Porter-Duff over compositing of multiple images, last input on top. Inputs must be "
        "premultiplied.",
        nlohmann::json({{"initial value", "vec4(0)"},
                        {"reduction", "current_value + accumulator * (1.0 - current_value.a)"}}));
    register_node<Reduce>(
        "Alpha Blend (Front-to-Back)",
        "Porter-Duff over compositing of multiple images with straight alpha, input 0 on top.",
        nlohmann::json(
            {{"initial value", "vec4(0)"},
             {"reduction",
              "vec4((accumulator.rgb * accumulator.a + current_value.rgb * current_value.a * "
              "(1.0 - accumulator.a)) / max(accumulator.a + current_value.a * (1.0 - "
              "accumulator.a), 1e-8), accumulator.a + current_value.a * (1.0 - accumulator.a))"}}));
    register_node<Reduce>(
        "Alpha Blend (Back-to-Front)",
        "Porter-Duff over compositing of multiple images with straight alpha, last input on top.",
        nlohmann::json(
            {{"initial value", "vec4(0)"},
             {"reduction",
              "vec4((current_value.rgb * current_value.a + accumulator.rgb * accumulator.a * "
              "(1.0 - current_value.a)) / max(current_value.a + accumulator.a * (1.0 - "
              "current_value.a), 1e-8), current_value.a + accumulator.a * (1.0 - "
              "current_value.a))"}}));
}

} // namespace merian
