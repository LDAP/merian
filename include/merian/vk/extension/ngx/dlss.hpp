#pragma once

#include "merian/utils/vector_matrix.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/extension/ngx/extension_ngx.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>
#include <string>
#include <vector>

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;

namespace merian {

// Ratio between target and render resolution, and with it the model NGX picks.
enum class DLSSQuality : uint32_t {
    UltraPerformance = 0,
    Performance,
    Balanced,
    Quality,
    DLAA,
};

inline const std::vector<std::string> DLSS_QUALITY_NAMES = {
    "ultra performance", "performance", "balanced", "quality", "DLAA",
};

enum class DLSSPreset : uint32_t {
    Default = 0,
    D = 4,
    E = 5,
    F = 6,
    J = 10,
    K = 11,
    L = 12,
    M = 13,
};

struct DLSSResolution {
    vk::Extent2D optimal;
    vk::Extent2D min;
    vk::Extent2D max;
};

struct DLSSCreateInfo {
    vk::Extent2D render_extent;
    vk::Extent2D target_extent;
    DLSSQuality quality = DLSSQuality::Quality;
    DLSSPreset preset = DLSSPreset::Default;
};

// All inputs are read in eGeneral. Guides and color at render resolution.
struct DLSSEvalInfo {
    // linear HDR
    ImageViewHandle color;
    // Ray reconstruction: view space depth. Super resolution: projected depth, 0 at the near plane.
    ImageViewHandle depth;
    // Pixels from this frame to the previous one, in the first two channels.
    ImageViewHandle motion_vectors;
    // Target resolution.
    ImageViewHandle output;

    // Sub-pixel offset the render camera was jittered by, in render pixels, x right and y down.
    float2 jitter{};
    // Drop the temporal history, for a cut or a teleport.
    bool reset = false;

    // Ray reconstruction only.
    ImageViewHandle diffuse_albedo;
    ImageViewHandle specular_albedo;
    // normal, linear roughness
    ImageViewHandle normal_roughness;
    // Optional: world space distance from the primary hit to the specular hit.
    ImageViewHandle specular_hit_distance;
    // Optional, one channel in [-1, 1]: positive follows the input over the history.
    ImageViewHandle responsivity;
    float4x4 world_to_view = identity();
    float4x4 view_to_clip = identity();
};

class ExtensionDLSS;

// One instantiated DLSS model, tied to the resolution pair it was created with.
class DLSS {
    friend ExtensionDLSS;

  public:
    void evaluate(const CommandBufferHandle& cmd, const DLSSEvalInfo& eval_info);

  private:
    DLSS(const std::shared_ptr<ExtensionNGX>& ngx,
         bool ray_reconstruction,
         const DLSSCreateInfo& create_info,
         const CommandBufferHandle& cmd);

    void evaluate_super_resolution(const CommandBufferHandle& cmd, const DLSSEvalInfo& eval_info);

    void evaluate_ray_reconstruction(const CommandBufferHandle& cmd, const DLSSEvalInfo& eval_info);

    const std::shared_ptr<ExtensionNGX> ngx;
    const bool ray_reconstruction;
    const DLSSCreateInfo create_info;

    struct ParameterDeleter {
        void operator()(NVSDK_NGX_Parameter* params) const;
    };
    struct FeatureDeleter {
        void operator()(NVSDK_NGX_Handle* handle) const;
    };

    std::unique_ptr<NVSDK_NGX_Parameter, ParameterDeleter> params;
    std::unique_ptr<NVSDK_NGX_Handle, FeatureDeleter> handle;
};

using DLSSHandle = std::shared_ptr<DLSS>;

} // namespace merian
