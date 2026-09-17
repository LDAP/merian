#include "merian/vk/extension/ngx/dlss.hpp"

#ifdef MERIAN_NGX_ENABLED
#include "merian/vk/utils/subresource_ranges.hpp"
#include "ngx.hpp"

#include <spdlog/spdlog.h>

#include <array>
#include <cassert>
#include <cstring>
#endif

namespace merian {

#ifdef MERIAN_NGX_ENABLED

namespace {

constexpr int FEATURE_CREATE_FLAGS =
    NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;

constexpr int SUPER_RESOLUTION_CREATE_FLAGS =
    FEATURE_CREATE_FLAGS | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

constexpr std::array<const char*, 6> SUPER_RESOLUTION_PRESET_HINTS = {
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
};

constexpr std::array<const char*, 6> RAY_RECONSTRUCTION_PRESET_HINTS = {
    NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
    NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality,
    NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
    NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
    NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
    NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
};

NVSDK_NGX_Resource_VK image_resource(const ImageViewHandle& view, const bool read_write) {
    const Image& image = *view->get_image();
    return NVSDK_NGX_Create_ImageView_Resource_VK(
        view->get_view(), image.get_image(), first_level_and_layer(),
        static_cast<VkFormat>(image.get_format()), image.get_extent().width,
        image.get_extent().height, read_write);
}

void throw_if_failed(const NVSDK_NGX_Result result, const std::string& what) {
    if (NVSDK_NGX_FAILED(result)) {
        throw MerianException{fmt::format("{}: {}", what, ngx_result_string(result))};
    }
}

// NGX multiplies row major matrices from the left (DLSS-RR Integration Guide 3.4.9), expects a
// left handed view space and takes mutable pointers.
struct NGXMatrices {
    std::array<float, 16> world_to_view;
    std::array<float, 16> view_to_clip;

    explicit NGXMatrices(const DLSSEvalInfo& eval_info) {
        const float4x4 mirror_z = scale(float3(1.f, 1.f, -1.f));
        const float4x4 view = transpose(mul(mirror_z, eval_info.world_to_view));
        const float4x4 clip = transpose(mul(eval_info.view_to_clip, mirror_z));
        std::memcpy(world_to_view.data(), value_ptr(view), sizeof(world_to_view));
        std::memcpy(view_to_clip.data(), value_ptr(clip), sizeof(view_to_clip));
    }
};

} // namespace

DLSS::DLSS(const std::shared_ptr<ExtensionNGX>& ngx,
           const bool ray_reconstruction,
           const DLSSCreateInfo& create_info,
           const CommandBufferHandle& cmd)
    : ngx(ngx), ray_reconstruction(ray_reconstruction), create_info(create_info) {
    assert(ngx->get_capability_parameters() != nullptr && "NGX is not initialized");
    const vk::Device device = ngx->get_device()->get_device();

    NVSDK_NGX_Parameter* allocated_params = nullptr;
    throw_if_failed(NVSDK_NGX_VULKAN_AllocateParameters(&allocated_params),
                    "could not allocate the DLSS parameters");
    params.reset(allocated_params);
    for (const char* hint :
         ray_reconstruction ? RAY_RECONSTRUCTION_PRESET_HINTS : SUPER_RESOLUTION_PRESET_HINTS) {
        NVSDK_NGX_Parameter_SetUI(params.get(), hint,
                                  static_cast<unsigned int>(create_info.preset));
    }

    NVSDK_NGX_Handle* feature = nullptr;

    if (ray_reconstruction) {
        NVSDK_NGX_DLSSD_Create_Params params_dlssd{};
        params_dlssd.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        params_dlssd.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
        params_dlssd.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
        params_dlssd.InWidth = create_info.render_extent.width;
        params_dlssd.InHeight = create_info.render_extent.height;
        params_dlssd.InTargetWidth = create_info.target_extent.width;
        params_dlssd.InTargetHeight = create_info.target_extent.height;
        params_dlssd.InPerfQualityValue = ngx_perf_quality(create_info.quality);
        params_dlssd.InFeatureCreateFlags = FEATURE_CREATE_FLAGS;
        throw_if_failed(NGX_VULKAN_CREATE_DLSSD_EXT1(device, cmd->get_command_buffer(), 1, 1,
                                                     &feature, params.get(), &params_dlssd),
                        "could not create the DLSS ray reconstruction feature");
    } else {
        NVSDK_NGX_DLSS_Create_Params params_dlss{};
        params_dlss.Feature.InWidth = create_info.render_extent.width;
        params_dlss.Feature.InHeight = create_info.render_extent.height;
        params_dlss.Feature.InTargetWidth = create_info.target_extent.width;
        params_dlss.Feature.InTargetHeight = create_info.target_extent.height;
        params_dlss.Feature.InPerfQualityValue = ngx_perf_quality(create_info.quality);
        params_dlss.InFeatureCreateFlags = SUPER_RESOLUTION_CREATE_FLAGS;
        throw_if_failed(NGX_VULKAN_CREATE_DLSS_EXT1(device, cmd->get_command_buffer(), 1, 1,
                                                    &feature, params.get(), &params_dlss),
                        "could not create the DLSS super resolution feature");
    }

    handle.reset(feature);

    SPDLOG_DEBUG("DLSS: {} {}x{} -> {}x{}",
                 ray_reconstruction ? "ray reconstruction" : "super resolution",
                 create_info.render_extent.width, create_info.render_extent.height,
                 create_info.target_extent.width, create_info.target_extent.height);
}

void DLSS::ParameterDeleter::operator()(NVSDK_NGX_Parameter* params) const {
    NVSDK_NGX_VULKAN_DestroyParameters(params);
}

void DLSS::FeatureDeleter::operator()(NVSDK_NGX_Handle* handle) const {
    NVSDK_NGX_VULKAN_ReleaseFeature(handle);
}

void DLSS::evaluate(const CommandBufferHandle& cmd, const DLSSEvalInfo& eval_info) {
    if (ray_reconstruction) {
        evaluate_ray_reconstruction(cmd, eval_info);
    } else {
        evaluate_super_resolution(cmd, eval_info);
    }
}

void DLSS::evaluate_super_resolution(const CommandBufferHandle& cmd,
                                     const DLSSEvalInfo& eval_info) {
    NVSDK_NGX_Resource_VK color = image_resource(eval_info.color, false);
    NVSDK_NGX_Resource_VK depth = image_resource(eval_info.depth, false);
    NVSDK_NGX_Resource_VK motion_vectors = image_resource(eval_info.motion_vectors, false);
    NVSDK_NGX_Resource_VK output = image_resource(eval_info.output, true);

    NVSDK_NGX_VK_DLSS_Eval_Params eval{};
    eval.Feature.pInColor = &color;
    eval.Feature.pInOutput = &output;
    eval.pInDepth = &depth;
    eval.pInMotionVectors = &motion_vectors;
    // NGX takes the projection jitter, which moves the image opposite to the sample (DLSS
    // Programming Guide 3.7.3).
    eval.InJitterOffsetX = -eval_info.jitter.x;
    eval.InJitterOffsetY = -eval_info.jitter.y;
    eval.InReset = static_cast<int>(eval_info.reset);
    eval.InRenderSubrectDimensions = {create_info.render_extent.width,
                                      create_info.render_extent.height};

    throw_if_failed(
        NGX_VULKAN_EVALUATE_DLSS_EXT(cmd->get_command_buffer(), handle.get(), params.get(), &eval),
        "DLSS super resolution failed");
}

void DLSS::evaluate_ray_reconstruction(const CommandBufferHandle& cmd,
                                       const DLSSEvalInfo& eval_info) {
    NVSDK_NGX_Resource_VK color = image_resource(eval_info.color, false);
    NVSDK_NGX_Resource_VK depth = image_resource(eval_info.depth, false);
    NVSDK_NGX_Resource_VK motion_vectors = image_resource(eval_info.motion_vectors, false);
    NVSDK_NGX_Resource_VK output = image_resource(eval_info.output, true);
    NVSDK_NGX_Resource_VK diffuse_albedo = image_resource(eval_info.diffuse_albedo, false);
    NVSDK_NGX_Resource_VK specular_albedo = image_resource(eval_info.specular_albedo, false);
    NVSDK_NGX_Resource_VK normal_roughness = image_resource(eval_info.normal_roughness, false);
    NVSDK_NGX_Resource_VK specular_hit_distance{};
    if (eval_info.specular_hit_distance) {
        specular_hit_distance = image_resource(eval_info.specular_hit_distance, false);
    }
    NVSDK_NGX_Resource_VK responsivity{};
    if (eval_info.responsivity) {
        responsivity = image_resource(eval_info.responsivity, false);
    }
    NGXMatrices matrices(eval_info);

    NVSDK_NGX_VK_DLSSD_Eval_Params eval{};
    eval.pInColor = &color;
    eval.pInOutput = &output;
    eval.pInDepth = &depth;
    eval.pInMotionVectors = &motion_vectors;
    eval.pInDiffuseAlbedo = &diffuse_albedo;
    eval.pInSpecularAlbedo = &specular_albedo;
    eval.pInNormals = &normal_roughness;
    eval.pInSpecularHitDistance =
        eval_info.specular_hit_distance ? &specular_hit_distance : nullptr;
    eval.pInResponsivityMask = eval_info.responsivity ? &responsivity : nullptr;
    eval.pInWorldToViewMatrix = matrices.world_to_view.data();
    eval.pInViewToClipMatrix = matrices.view_to_clip.data();
    eval.InJitterOffsetX = -eval_info.jitter.x;
    eval.InJitterOffsetY = -eval_info.jitter.y;
    eval.InReset = static_cast<int>(eval_info.reset);
    eval.InRenderSubrectDimensions = {create_info.render_extent.width,
                                      create_info.render_extent.height};

    throw_if_failed(
        NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd->get_command_buffer(), handle.get(), params.get(), &eval),
        "DLSS ray reconstruction failed");
}

#else

void DLSS::evaluate([[maybe_unused]] const CommandBufferHandle& cmd,
                    [[maybe_unused]] const DLSSEvalInfo& eval_info) {
    throw MerianException{"merian was built without the NGX SDK (dlss option)"};
}

#endif

} // namespace merian
