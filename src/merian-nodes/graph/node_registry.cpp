#include "merian-nodes/graph/node_registry.hpp"

#include "merian-nodes/nodes/ab_compare/ab_compare.hpp"
#include "merian-nodes/nodes/accumulate/accumulate.hpp"
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
#include "merian-nodes/nodes/reduce/reduce.hpp"
#include "merian-nodes/nodes/shadertoy/shadertoy.hpp"
#include "merian-nodes/nodes/slang_compute/slang_compute.hpp"
#include "merian-nodes/nodes/svgf/svgf.hpp"
#include "merian-nodes/nodes/taa/taa.hpp"
#include "merian-nodes/nodes/tonemap/tonemap.hpp"
#include "merian-nodes/nodes/vkdt_filmcurv/vkdt_filmcurv.hpp"

namespace merian_nodes {

NodeRegistry::NodeRegistry(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    register_node_type<ABSplit>(NodeTypeInfo{"AB Split", "Compare two inputs in a split-view.",
                                             []() { return std::make_shared<ABSplit>(); }});
    register_node_type<ABSideBySide>(
        NodeTypeInfo{"AB Side By Side", "Compare two inputs in a side by side view.",
                     []() { return std::make_shared<ABSideBySide>(); }});
    register_node_type<Accumulate>(
        NodeTypeInfo{"Accumulate", "Accumulate values across multiple iterations.",
                     [=]() { return std::make_shared<Accumulate>(context, allocator); }});
    register_node_type<DeviceASBuilder>(
        NodeTypeInfo{"Acceleration Structure Builder",
                     "Build acceleration structures from geometry on the device.",
                     [=]() { return std::make_shared<DeviceASBuilder>(context, allocator); }});
    register_node_type<Bloom>(NodeTypeInfo{"Bloom",
                                           "Selectively blurs pixels that surpass a threshold.",
                                           [=]() { return std::make_shared<Bloom>(context); }});
    register_node_type<ColorImage>(
        NodeTypeInfo{"Color", "Outputs a image filled cleared with the selected color.",
                     [=]() { return std::make_shared<ColorImage>(); }});
    register_node_type<AutoExposure>(NodeTypeInfo{
        "Exposure", "Exposure with camera-like controls. Includes a robust auto-exposure mode",
        [=]() { return std::make_shared<AutoExposure>(context); }});
    register_node_type<FXAA>(
        NodeTypeInfo{"FXAA",
                     "Fast approximate anti-aliasing (FXAA) is a screen-space "
                     "anti-aliasing algorithm created by Timothy Lottes at NVIDIA.",
                     [=]() { return std::make_shared<FXAA>(context); }});
    register_node_type<GLFWWindow>(
        NodeTypeInfo{"Window (GLFW)", "Outputs to a window created with GLFW.",
                     [=]() { return std::make_shared<GLFWWindow>(context); }});
    register_node_type<HDRImageRead>(
        NodeTypeInfo{"HDR Image", "Loads an HDR image.",
                     [=]() { return std::make_shared<HDRImageRead>(context); }});
    register_node_type<LDRImageRead>(
        NodeTypeInfo{"LDR Image", "Loads a LDR image.",
                     [=]() { return std::make_shared<LDRImageRead>(context); }});
    register_node_type<ImageWrite>(
        NodeTypeInfo{"Image Write", "Stores a graph output as image file.",
                     [=]() { return std::make_shared<ImageWrite>(context, allocator); }});
    register_node_type<MeanToBuffer>(NodeTypeInfo{
        "Mean", "Computes the mean of an image and outputs it as a single buffer element.",
        [=]() { return std::make_shared<MeanToBuffer>(context); }});
    register_node_type<MedianApproxNode>(NodeTypeInfo{
        "Median (Approximation)", "Computes an approximation of the median of a component.",
        [=]() { return std::make_shared<MedianApproxNode>(context); }});
    register_node_type<Reduce>(NodeTypeInfo{"Reduce", "Reduce values of multiple input images.",
                                            [=]() { return std::make_shared<Reduce>(context); }});
    register_node_type<Shadertoy>(
        NodeTypeInfo{"Shadertoy", "Execute Shadertoy-like shaders (Limited implementation).",
                     [=]() { return std::make_shared<Shadertoy>(context); }});
    register_node_type<SlangCompute>(
        NodeTypeInfo{"Slang Compute", "Execute Slang shaders with automatic layout reflection.",
                     [=]() { return std::make_shared<SlangCompute>(context); }});
    register_node_type<SVGF>(
        NodeTypeInfo{"Denoiser (SVGF)", "Spatiotemporal Variance-Guided Filtering.",
                     [=]() { return std::make_shared<SVGF>(context, allocator); }});
    register_node_type<TAA>(NodeTypeInfo{"TAA", "Temporal Anti-Aliasing.",
                                         [=]() { return std::make_shared<TAA>(context); }});
    register_node_type<Tonemap>(NodeTypeInfo{"Tonemap",
                                             "Convert a HDR image to LDR using various tonemaps.",
                                             [=]() { return std::make_shared<Tonemap>(context); }});
    register_node_type<VKDTFilmcurv>(
        NodeTypeInfo{"Curves", "Adjust brightness and contrast. Ported from VKDT.",
                     [=]() { return std::make_shared<VKDTFilmcurv>(context); }});

    register_node<Reduce>(
        "Add", "Add values of multiple input images.",
        nlohmann::json({{"initial value", ""}, {"initial value", "accumulator + current_value"}}));
    register_node<Reduce>(
        "Multiply", "Multiply values of multiple input images.",
        nlohmann::json({{"initial value", ""}, {"initial value", "accumulator * current_value"}}));
    register_node<Reduce>(
        "Divide", "Divide values of multiple input images.",
        nlohmann::json({{"initial value", ""}, {"initial value", "accumulator / current_value"}}));
    register_node<Reduce>(
        "Subtract", "Subtract values of multiple input images.",
        nlohmann::json({{"initial value", ""}, {"initial value", "accumulator - current_value"}}));
    register_node<Reduce>("Min", "Compute minimum over all input images.",
                          nlohmann::json({{"initial value", ""},
                                          {"initial value", "min(accumulator, current_value)"}}));
    register_node<Reduce>("Max", "Compute maximum over all input images.",
                          nlohmann::json({{"initial value", ""},
                                          {"initial value", "max(accumulator, current_value)"}}));
}

} // namespace merian_nodes
