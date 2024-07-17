#include "node_registry.hpp"

#include "merian-nodes/nodes/ab_compare/ab_compare.hpp"
#include "merian-nodes/nodes/accumulate/accumulate.hpp"
#include "merian-nodes/nodes/add/add.hpp"
#include "merian-nodes/nodes/as_builder/device_as_builder.hpp"
#include "merian-nodes/nodes/bloom/bloom.hpp"
#include "merian-nodes/nodes/color_image/color_output.hpp"
#include "merian-nodes/nodes/exposure/exposure.hpp"
#include "merian-nodes/nodes/fxaa/fxaa.hpp"
#include "merian-nodes/nodes/glfw_window/glfw_window.hpp"
#include "merian-nodes/nodes/image_read/hdr_image.hpp"
#include "merian-nodes/nodes/image_read/ldr_image.hpp"
#include "merian-nodes/nodes/image_write/image_write.hpp"
#include "merian-nodes/nodes/mean/mean.hpp"
#include "merian-nodes/nodes/median_approx/median.hpp"
#include "merian-nodes/nodes/svgf/svgf.hpp"
#include "merian-nodes/nodes/taa/taa.hpp"
#include "merian-nodes/nodes/tonemap/tonemap.hpp"
#include "merian-nodes/nodes/vkdt_filmcurv/vkdt_filmcurv.hpp"

namespace merian_nodes {

NodeRegistry::NodeRegistry(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    register_node<ABSplit>(NodeInfo{"AB Split", "Compare two inputs in a split-view.",
                                    []() { return std::make_shared<ABSplit>(); }});
    register_node<ABSideBySide>(NodeInfo{"AB Side By Side",
                                         "Compare two inputs in a side by side view.",
                                         []() { return std::make_shared<ABSideBySide>(); }});
    register_node<Accumulate>(
        NodeInfo{"Accumulate", "Accumulate values across multiple iterations.",
                 [=]() { return std::make_shared<Accumulate>(context, allocator); }});
    register_node<Add>(NodeInfo{"Add", "Add values of two images.",
                                [=]() { return std::make_shared<Add>(context); }});
    register_node<DeviceASBuilder>(
        NodeInfo{"Acceleration Structure Builder",
                 "Build acceleration structures from geometry on the device.",
                 [=]() { return std::make_shared<DeviceASBuilder>(context, allocator); }});
    register_node<Bloom>(NodeInfo{"Bloom", "Selectively blurs pixels that surpass a threshold.",
                                  [=]() { return std::make_shared<Bloom>(context); }});
    register_node<ColorImage>(NodeInfo{"Color",
                                       "Outputs a image filled cleared with the selected color.",
                                       [=]() { return std::make_shared<ColorImage>(); }});
    register_node<AutoExposure>(NodeInfo{
        "Exposure", "Exposure with camera-like controls. Includes a robust auto-exposure mode",
        [=]() { return std::make_shared<AutoExposure>(context); }});
    register_node<FXAA>(NodeInfo{"FXAA",
                                 "Fast approximate anti-aliasing (FXAA) is a screen-space "
                                 "anti-aliasing algorithm created by Timothy Lottes at NVIDIA.",
                                 [=]() { return std::make_shared<FXAA>(context); }});
    register_node<GLFWWindow>(NodeInfo{"Window (GLFW)", "Outputs to a window created with GLFW.",
                                       [=]() { return std::make_shared<GLFWWindow>(context); }});
    register_node<HDRImageRead>(NodeInfo{"HDR Image", "Loads an HDR image.", [=]() {
                                             return std::make_shared<HDRImageRead>(context);
                                         }});
    register_node<LDRImageRead>(NodeInfo{"LDR Image", "Loads a LDR image.", [=]() {
                                             return std::make_shared<LDRImageRead>(context);
                                         }});
    register_node<ImageWrite>(
        NodeInfo{"Image Write", "Stores a graph output as image file.",
                 [=]() { return std::make_shared<ImageWrite>(context, allocator); }});
    register_node<MeanToBuffer>(
        NodeInfo{"Mean", "Computes the mean of an image and outputs it as a single buffer element.",
                 [=]() { return std::make_shared<MeanToBuffer>(context); }});
    register_node<MedianApproxNode>(NodeInfo{
        "Median (Approximation)", "Computes an approximation of the median of a component.",
        [=]() { return std::make_shared<MedianApproxNode>(context); }});
    // TODO: Shadertoy
    register_node<SVGF>(NodeInfo{"Denoiser (SVGF)", "Spatiotemporal Variance-Guided Filtering.",
                                 [=]() { return std::make_shared<SVGF>(context, allocator); }});
    register_node<TAA>(NodeInfo{"TAA", "Temporal Anti-Aliasing.",
                                [=]() { return std::make_shared<TAA>(context); }});
    register_node<Tonemap>(NodeInfo{"Tonemap", "Convert a HDR image to LDR using various tonemaps.",
                                    [=]() { return std::make_shared<Tonemap>(context); }});
    register_node<VKDTFilmcurv>(
        NodeInfo{"Curves", "Adjust brightness and contrast. Ported from VKDT.",
                 [=]() { return std::make_shared<VKDTFilmcurv>(context); }});
}

} // namespace merian_nodes
