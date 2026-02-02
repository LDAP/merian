#include "merian/vk/command/command_buffer.hpp"

#include <spdlog/spdlog.h>

namespace merian {

CommandBuffer::CommandBuffer(const CommandPoolHandle& pool, const vk::CommandBufferLevel level)
    : pool(pool) {

    const vk::CommandBufferAllocateInfo info{*pool, level, 1};
    check_result(pool->get_context()->get_device()->get_device().allocateCommandBuffers(&info, &cmd),
                 "could not allocate command buffer");

    SPDLOG_DEBUG("allocate command buffer ({})", fmt::ptr(static_cast<VkCommandBuffer>(cmd)));
}

CommandBuffer::~CommandBuffer() {
    pool->get_context()->get_device()->get_device().freeCommandBuffers(*pool, cmd);

    SPDLOG_DEBUG("free command buffer ({})", fmt::ptr(static_cast<VkCommandBuffer>(cmd)));
}

void CommandBuffer::begin(const vk::CommandBufferBeginInfo& info) {
    cmd.begin(info);
    descriptor_buffers_need_rebind = false;

#ifndef NDEBUG
    did_bind_descriptor_buffers = false;
#endif
}

void CommandBuffer::begin(const vk::CommandBufferUsageFlags flags,
                          const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    begin(vk::CommandBufferBeginInfo{flags, pInheritanceInfo});
}

void CommandBuffer::end() {
    cmd.end();
    current_pipeline.reset();
    descriptor_buffer_bindings.clear();
    pipeline_descriptor_buffer_set_offsets.clear();
}

void CommandBuffer::begin_render_pass(const FramebufferHandle& framebuffer,
                                      const vk::Rect2D& render_area,
                                      const vk::ArrayProxy<const vk::ClearValue>& clear_values,
                                      const vk::SubpassContents subpass_contents) {
    vk::RenderPassBeginInfo begin_info{*framebuffer->get_renderpass(), *framebuffer, render_area,
                                       clear_values};
    cmd.beginRenderPass(begin_info, subpass_contents);
    keep_until_pool_reset(framebuffer);
}

void CommandBuffer::begin_render_pass(const FramebufferHandle& framebuffer,
                                      const vk::ArrayProxy<const vk::ClearValue>& clear_values,
                                      const vk::SubpassContents subpass_contents) {
    begin_render_pass(framebuffer, vk::Rect2D{{}, framebuffer->get_extent()}, clear_values,
                      subpass_contents);
}

void CommandBuffer::copy(const BufferHandle& src_buffer,
                         const BufferHandle& dst_buffer,
                         const vk::ArrayProxy<const vk::BufferCopy>& regions) {
    cmd.copyBuffer(*src_buffer, *dst_buffer, regions);
    keep_until_pool_reset(src_buffer);
    keep_until_pool_reset(dst_buffer);
}

void CommandBuffer::fill(const BufferHandle& buffer, const uint32_t data) {
    cmd.fillBuffer(*buffer, 0, VK_WHOLE_SIZE, data);
    keep_until_pool_reset(buffer);
}

void CommandBuffer::copy(const ImageHandle& src_image,
                         const vk::ImageLayout src_layout,
                         const ImageHandle& dst_image,
                         const vk::ImageLayout dst_layout,
                         const vk::ArrayProxy<const vk::ImageCopy>& regions) {
    cmd.copyImage(*src_image, src_layout, *dst_image, dst_layout, regions);
    keep_until_pool_reset(src_image);
    keep_until_pool_reset(dst_image);
}

void CommandBuffer::copy(const ImageHandle& src_image,
                         const ImageHandle& dst_image,
                         const vk::ArrayProxy<const vk::ImageCopy>& regions) {
    copy(src_image, src_image->get_current_layout(), dst_image, dst_image->get_current_layout(),
         regions);
}

void CommandBuffer::copy(const ImageHandle& src_image, const ImageHandle& dst_image) {
    copy(src_image, src_image->get_current_layout(), dst_image, dst_image->get_current_layout(),
         vk::ImageCopy{all_layers(), {}, all_layers(), {}, src_image->get_extent()});
}

void CommandBuffer::blit(const ImageHandle& src_image,
                         const vk::ImageLayout src_layout,
                         const ImageHandle& dst_image,
                         const vk::ImageLayout dst_layout,
                         const vk::ArrayProxy<const vk::ImageBlit>& regions,
                         const vk::Filter filter) {
    cmd.blitImage(*src_image, src_layout, *dst_image, dst_layout, regions, filter);
    keep_until_pool_reset(src_image);
    keep_until_pool_reset(dst_image);
}

void CommandBuffer::blit(const ImageHandle& src_image,
                         const ImageHandle& dst_image,
                         const vk::ArrayProxy<const vk::ImageBlit>& regions,
                         const vk::Filter filter) {
    blit(src_image, src_image->get_current_layout(), dst_image, dst_image->get_current_layout(),
         regions, filter);
}

void CommandBuffer::clear(const ImageHandle& image,
                          const vk::ImageLayout layout,
                          const vk::ClearColorValue color,
                          const vk::ArrayProxy<vk::ImageSubresourceRange>& ranges) {
    cmd.clearColorImage(*image, layout, color, ranges);
    keep_until_pool_reset(image);
}

void CommandBuffer::clear(const ImageHandle& image,
                          const vk::ClearColorValue color,
                          const vk::ArrayProxy<vk::ImageSubresourceRange>& ranges) {
    clear(image, image->get_current_layout(), color, ranges);
}

void CommandBuffer::copy(const ImageHandle& src_image,
                         const vk::ImageLayout src_layout,
                         const BufferHandle& dst_buffer,
                         const vk::ArrayProxy<const vk::BufferImageCopy>& regions) {
    cmd.copyImageToBuffer(*src_image, src_layout, *dst_buffer, regions);
    keep_until_pool_reset(src_image);
    keep_until_pool_reset(dst_buffer);
}

void CommandBuffer::copy(const ImageHandle& src_image,
                         const BufferHandle& dst_buffer,
                         const vk::ArrayProxy<const vk::BufferImageCopy>& regions) {
    copy(src_image, src_image->get_current_layout(), dst_buffer, regions);
}

void CommandBuffer::copy(const BufferHandle& src_buffer,
                         const ImageHandle& dst_image,
                         const vk::ImageLayout dst_layout,
                         const vk::ArrayProxy<const vk::BufferImageCopy>& regions) {
    cmd.copyBufferToImage(*src_buffer, *dst_image, dst_layout, regions);
    keep_until_pool_reset(src_buffer);
    keep_until_pool_reset(dst_image);
}

void CommandBuffer::copy(const BufferHandle& src_buffer,
                         const ImageHandle& dst_image,
                         const vk::ArrayProxy<const vk::BufferImageCopy>& regions) {
    copy(src_buffer, dst_image, dst_image->get_current_layout(), regions);
}

void CommandBuffer::push_descriptor_set(const PipelineHandle& pipeline,
                                        const uint32_t set,
                                        const vk::ArrayProxy<vk::WriteDescriptorSet>& writes) {
    cmd.pushDescriptorSetKHR(pipeline->get_pipeline_bind_point(), *pipeline->get_layout(), set,
                             writes);
    keep_until_pool_reset(pipeline);
}

void CommandBuffer::push_descriptor_set(const PipelineHandle& pipeline,
                                        const uint32_t set_index,
                                        const ConstPushDescriptorSetHandle& set) {
    push_descriptor_set(pipeline, set_index, set->get_writes());
    for (const auto& res : set->resources) {
        keep_until_pool_reset(res);
    }
}

void CommandBuffer::push_descriptor_set(const PipelineHandle& pipeline,
                                        const std::vector<vk::WriteDescriptorSet>& writes) {
    push_descriptor_set(pipeline, 0, writes);
}

void CommandBuffer::push_constant(const PipelineHandle& pipeline,
                                  const void* values,
                                  const uint32_t id) {
    auto range = pipeline->get_layout()->get_push_constant_range(id);
    push_constant(pipeline, range.stageFlags, range.offset, range.size, values);
}

void CommandBuffer::push_constant(const PipelineHandle& pipeline,
                                  const vk::ShaderStageFlags flags,
                                  const uint32_t offset,
                                  const uint32_t size,
                                  const void* values) {
    cmd.pushConstants(*pipeline->get_layout(), flags, offset, size, values);
    keep_until_pool_reset(pipeline);
}

void CommandBuffer::bind(const PipelineHandle& pipeline) {
    cmd.bindPipeline(pipeline->get_pipeline_bind_point(), *pipeline);
    keep_until_pool_reset(pipeline);

#ifndef NDEBUG
    static bool did_warn_descriptor_buffer = false;
    if (!did_warn_descriptor_buffer && current_pipeline &&
        current_pipeline->supports_descriptor_buffer() && pipeline->supports_descriptor_set()) {
        SPDLOG_WARN("Do not mix and match descriptor buffers and descriptor sets as this can "
                    "cause a ALL_COMMANDS -> ALL_COMMANDS pipeline barrier. See "
                    "https://www.khronos.org/blog/vk-ext-descriptor-buffer");
        did_warn_descriptor_buffer = true;
    }
#endif

    current_pipeline = pipeline;
}

void CommandBuffer::dispatch(const uint32_t group_count_x,
                             const uint32_t group_count_y,
                             const uint32_t group_count_z) {
    if (current_pipeline->supports_descriptor_buffer()) {
        update_descriptor_buffer_bindings(current_pipeline);
    }

    cmd.dispatch(group_count_x, group_count_y, group_count_z);
}

void CommandBuffer::dispatch(const vk::Extent3D& extent,
                             const uint32_t local_size_x,
                             const uint32_t local_size_y,
                             const uint32_t local_size_z) {
    dispatch((extent.width + local_size_x - 1) / local_size_x,
             (extent.height + local_size_y - 1) / local_size_y,
             (extent.depth + local_size_z - 1) / local_size_z);
}

void CommandBuffer::dispatch(const vk::Extent2D& extent,
                             const uint32_t local_size_x,
                             const uint32_t local_size_y) {
    dispatch((extent.width + local_size_x - 1) / local_size_x,
             (extent.height + local_size_y - 1) / local_size_y, 1);
}

void CommandBuffer::copy_acceleration_structure(const AccelerationStructureHandle& src,
                                                const AccelerationStructureHandle& dst,
                                                const vk::CopyAccelerationStructureModeKHR mode) {
    cmd.copyAccelerationStructureKHR(vk::CopyAccelerationStructureInfoKHR{*src, *dst, mode});
    keep_until_pool_reset(src);
    keep_until_pool_reset(dst);
}

} // namespace merian
