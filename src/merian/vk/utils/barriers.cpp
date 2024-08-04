#include "merian/vk/utils/barriers.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"
#include <vulkan/vulkan.hpp>

#include <unordered_map>

namespace merian {

vk::PipelineStageFlags pipeline_stage_for_access_flags(const vk::AccessFlags& flags) {
    using AF = vk::AccessFlagBits;
    using PS = vk::PipelineStageFlagBits;

    // clang-format off
    static std::unordered_map<vk::AccessFlagBits, vk::PipelineStageFlags> flag_map{
        {AF::eIndirectCommandRead                 , PS::eDrawIndirect}, 
        {AF::eIndexRead                           , PS::eVertexInput}, 
        {AF::eVertexAttributeRead                 , PS::eVertexInput}, 
        {AF::eUniformRead                         , all_shaders}, 
        {AF::eInputAttachmentRead                 , PS::eFragmentShader}, 
        {AF::eShaderRead                          , all_shaders}, 
        {AF::eShaderWrite                         , all_shaders}, 
        {AF::eColorAttachmentRead                 , PS::eColorAttachmentOutput}, 
        {AF::eColorAttachmentWrite                , PS::eColorAttachmentOutput}, 
        {AF::eDepthStencilAttachmentRead          , PS::eEarlyFragmentTests | PS::eLateFragmentTests}, 
        {AF::eDepthStencilAttachmentWrite         , PS::eEarlyFragmentTests | PS::eLateFragmentTests}, 
        {AF::eTransferRead                        , PS::eTransfer}, 
        {AF::eTransferWrite                       , PS::eTransfer}, 
        {AF::eHostRead                            , PS::eHost}, 
        {AF::eHostWrite                           , PS::eHost}, 
        {AF::eMemoryRead                          , {}}, 
        {AF::eMemoryWrite                         , {}}, 
        {AF::eNone                                , PS::eTopOfPipe}, 
        // {AF::eTransformFeedbackWriteEXT           , }, 
        // {AF::eTransformFeedbackCounterReadEXT     , }, 
        // {AF::eTransformFeedbackCounterWriteEXT    , }, 
        // {AF::eConditionalRenderingReadEXT         , }, 
        {AF::eColorAttachmentReadNoncoherentEXT   , PS::eColorAttachmentOutput}, 
        {AF::eAccelerationStructureReadKHR        , PS::eAccelerationStructureBuildNV | all_shaders}, 
        {AF::eAccelerationStructureWriteKHR       , PS::eAccelerationStructureBuildKHR}, 
        //{AF::eShadingRateImageReadNV              , }, 
        {AF::eAccelerationStructureReadNV         , PS::eAccelerationStructureBuildNV | all_shaders | PS::eRayTracingShaderNV}, 
        {AF::eAccelerationStructureWriteNV        , PS::eAccelerationStructureBuildNV}, 
        // {AF::eFragmentDensityMapReadEXT           , }, 
        // {AF::eFragmentShadingRateAttachmentReadKHR, }, 
        {AF::eCommandPreprocessReadNV             , PS::eCommandPreprocessNV}, 
        {AF::eCommandPreprocessWriteNV            , PS::eCommandPreprocessNV}, 
    };
    //clang-format on

    vk::PipelineStageFlags result_flags;
    for (auto& it : flag_map) {
        if ((it.first & flags) == it.first) {
            result_flags |= it.second;
        }
    }
    return result_flags;
}

vk::ImageMemoryBarrier barrier_image_layout(const vk::Image& image,
                              const vk::ImageLayout& old_image_layout,
                              const vk::ImageLayout& new_image_layout,
                              const vk::ImageSubresourceRange& subresource_range) {
    vk::ImageMemoryBarrier image_memory_barrier{
        access_flags_for_image_layout(old_image_layout),
        access_flags_for_image_layout(new_image_layout),
        old_image_layout,
        new_image_layout,
        // Fix for a validation issue - should be needed when vk::Image sharing mode is
        // VK_SHARING_MODE_EXCLUSIVE and the values of srcQueueFamilyIndex and dstQueueFamilyIndex
        // are equal, no ownership transfer is performed, and the barrier operates as if they were
        // both set to VK_QUEUE_FAMILY_IGNORED.
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image,
        subresource_range,
    };
    return image_memory_barrier;
}

void cmd_barrier_image_layout(const vk::CommandBuffer& cmd,
                              const vk::Image& image,
                              const vk::ImageLayout& old_image_layout,
                              const vk::ImageLayout& new_image_layout,
                              const vk::ImageSubresourceRange& subresource_range) {

    vk::ImageMemoryBarrier image_memory_barrier = barrier_image_layout(image, old_image_layout, new_image_layout, subresource_range);

    vk::PipelineStageFlags srcStageMask = pipeline_stage_for_image_layout(old_image_layout);
    vk::PipelineStageFlags dstStageMask = pipeline_stage_for_image_layout(new_image_layout);

    cmd.pipelineBarrier(srcStageMask, dstStageMask, {}, 0, nullptr, 0, nullptr, 1,
                              &image_memory_barrier);
}

vk::ImageMemoryBarrier barrier_image_layout(const vk::Image& image,
                              const vk::ImageLayout& old_image_layout,
                              const vk::ImageLayout& new_image_layout,
                              const vk::ImageAspectFlags& aspect_mask) {
    return barrier_image_layout(image, old_image_layout, new_image_layout, all_levels_and_layers(aspect_mask));
}

void cmd_barrier_image_layout(const vk::CommandBuffer& cmd,
                              const vk::Image& image,
                              const vk::ImageLayout& old_image_layout,
                              const vk::ImageLayout& new_image_layout,
                              const vk::ImageAspectFlags& aspect_mask) {
    cmd_barrier_image_layout(cmd, image, old_image_layout, new_image_layout, all_levels_and_layers(aspect_mask));
}

} // namespace merian
