#include "merian/vk/utils/barriers.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

vk::PipelineStageFlags pipeline_stage_for_access_flags(vk::AccessFlags flags) {
    using AF = vk::AccessFlagBits;
    using PS = vk::PipelineStageFlagBits;
    static vk::PipelineStageFlags shaders =
        PS::eVertexShader | PS::eTessellationControlShader | PS::eTessellationEvaluationShader |
        PS::eGeometryShader | PS::eFragmentShader | PS::eComputeShader | PS::eRayTracingShaderKHR;

    // clang-format off
    static std::unordered_map<vk::AccessFlagBits, vk::PipelineStageFlags> flag_map{
        {AF::eIndirectCommandRead                 , PS::eDrawIndirect}, 
        {AF::eIndexRead                           , PS::eVertexInput}, 
        {AF::eVertexAttributeRead                 , PS::eVertexInput}, 
        {AF::eUniformRead                         , shaders}, 
        {AF::eInputAttachmentRead                 , PS::eFragmentShader}, 
        {AF::eShaderRead                          , shaders}, 
        {AF::eShaderWrite                         , shaders}, 
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
        {AF::eAccelerationStructureReadKHR        , PS::eAccelerationStructureBuildNV | shaders}, 
        {AF::eAccelerationStructureWriteKHR       , PS::eAccelerationStructureBuildKHR}, 
        //{AF::eShadingRateImageReadNV              , }, 
        {AF::eAccelerationStructureReadNV         , PS::eAccelerationStructureBuildNV | shaders | PS::eRayTracingShaderNV}, 
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

vk::ImageMemoryBarrier barrier_image_layout(vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
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

void cmd_barrier_image_layout(vk::CommandBuffer cmd,
                              vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              const vk::ImageSubresourceRange& subresource_range) {

    vk::ImageMemoryBarrier image_memory_barrier = barrier_image_layout(image, old_image_layout, new_image_layout, subresource_range);

    vk::PipelineStageFlags srcStageMask = pipeline_stage_for_image_layout(old_image_layout);
    vk::PipelineStageFlags destStageMask = pipeline_stage_for_image_layout(new_image_layout);

    cmd.pipelineBarrier(srcStageMask, destStageMask, {}, 0, nullptr, 0, nullptr, 1,
                              &image_memory_barrier);
}

vk::ImageMemoryBarrier barrier_image_layout(vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              vk::ImageAspectFlags aspect_mask) {
    vk::ImageSubresourceRange subresourceRange{
        aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS,
    };
    return barrier_image_layout(image, old_image_layout, new_image_layout, subresourceRange);
}

void cmd_barrier_image_layout(vk::CommandBuffer cmd,
                              vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              vk::ImageAspectFlags aspect_mask) {
    vk::ImageSubresourceRange subresourceRange{
        aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS,
    };

    cmd_barrier_image_layout(cmd, image, old_image_layout, new_image_layout, subresourceRange);
}

// A barrier between compute shader write and host read
void cmd_barrier_compute_host(const vk::CommandBuffer cmd) {
    vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost,
                        {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

} // namespace merian
