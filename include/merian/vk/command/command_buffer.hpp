#pragma once

#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/command/event.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/renderpass/framebuffer.hpp"
#include "merian/vk/utils/check_result.hpp"
#include "merian/vk/utils/query_pool.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

class CommandBuffer;
using CommandBufferHandle = std::shared_ptr<CommandBuffer>;

/**
 * @brief      Abstraction for command buffers which ensures objects are not destroyed until after
 * command buffer execution.
 */
class CommandBuffer {
  public:
    friend class CommandPool;

  public:
    CommandBuffer() = delete;
    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer(const CommandBuffer&&) = delete;

    CommandBuffer(const CommandPoolHandle& pool,
                  const vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary)
        : pool(pool) {

        const vk::CommandBufferAllocateInfo info{*pool, level, 1};
        check_result(pool->get_context()->device.allocateCommandBuffers(&info, &cmd),
                     "could not allocate command buffer");

        SPDLOG_DEBUG("allocate command buffer ({})", fmt::ptr(static_cast<VkCommandBuffer>(cmd)));
    }

    ~CommandBuffer() {
        pool->get_context()->device.freeCommandBuffers(*pool, cmd);

        SPDLOG_DEBUG("free command buffer ({})", fmt::ptr(static_cast<VkCommandBuffer>(cmd)));
    }

    // ------------------------------------------------------------

    void begin(const vk::CommandBufferBeginInfo& info) {
        cmd.begin(info);
    }

    void
    begin(const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
          const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        begin(vk::CommandBufferBeginInfo{flags, pInheritanceInfo});
    }

    void end() {
        cmd.end();
    }

    // ------------------------------------------------------------

    operator const vk::CommandBuffer&() const noexcept {
        return cmd;
    }

    const vk::CommandBuffer& operator*() const noexcept {
        return cmd;
    }

    const vk::CommandBuffer& get_command_buffer() const noexcept {
        return cmd;
    }

    const CommandPoolHandle& get_pool() const {
        return pool;
    }

    // ------------------------------------------------------------

    void keep_until_pool_reset(const ObjectHandle& object) {
        pool->keep_until_pool_reset(object);
    }

    // ------------------------------------------------------------
    // FRAMEBUFFER / RENDERPASS

    void
    begin_render_pass(const FramebufferHandle& framebuffer,
                      const vk::Rect2D& render_area,
                      const vk::ArrayProxy<const vk::ClearValue>& clear_values = {},
                      const vk::SubpassContents subpass_contents = vk::SubpassContents::eInline) {
        vk::RenderPassBeginInfo begin_info{*framebuffer->get_renderpass(), *framebuffer,
                                           render_area, clear_values};
        cmd.beginRenderPass(begin_info, subpass_contents);
        keep_until_pool_reset(framebuffer);
    }

    void
    begin_render_pass(const FramebufferHandle& framebuffer,
                      const vk::ArrayProxy<const vk::ClearValue>& clear_values = {},
                      const vk::SubpassContents subpass_contents = vk::SubpassContents::eInline) {
        begin_render_pass(framebuffer, vk::Rect2D{{}, framebuffer->get_extent()}, clear_values,
                          subpass_contents);
    }

    void end_render_pass() {
        cmd.endRenderPass();
    }

    // ------------------------------------------------------------
    // BUFFER

    void copy(const BufferHandle& src_buffer,
              const BufferHandle& dst_buffer,
              const vk::ArrayProxy<const vk::BufferCopy>& regions) {
        cmd.copyBuffer(*src_buffer, *dst_buffer, regions);
        keep_until_pool_reset(src_buffer);
        keep_until_pool_reset(dst_buffer);
    }

    void fill(const BufferHandle& buffer, const uint32_t data = 0) {
        cmd.fillBuffer(*buffer, 0, VK_WHOLE_SIZE, data);
        keep_until_pool_reset(buffer);
    }

    // ------------------------------------------------------------
    // IMAGE

    void copy(const ImageHandle& src_image,
              const vk::ImageLayout src_layout,
              const ImageHandle& dst_image,
              const vk::ImageLayout dst_layout,
              const vk::ArrayProxy<const vk::ImageCopy>& regions) {
        cmd.copyImage(*src_image, src_layout, *dst_image, dst_layout, regions);
        keep_until_pool_reset(src_image);
        keep_until_pool_reset(dst_image);
    }

    void copy(const ImageHandle& src_image,
              const ImageHandle& dst_image,
              const vk::ArrayProxy<const vk::ImageCopy>& regions) {
        copy(src_image, src_image->get_current_layout(), dst_image, dst_image->get_current_layout(),
             regions);
    }

    void copy(const ImageHandle& src_image, const ImageHandle& dst_image) {
        copy(src_image, src_image->get_current_layout(), dst_image, dst_image->get_current_layout(),
             vk::ImageCopy{all_layers(), {}, all_layers(), {}, src_image->get_extent()});
    }

    void blit(const ImageHandle& src_image,
              const vk::ImageLayout src_layout,
              const ImageHandle& dst_image,
              const vk::ImageLayout dst_layout,
              const vk::ArrayProxy<const vk::ImageBlit>& regions,
              const vk::Filter filter = vk::Filter::eLinear) {
        cmd.blitImage(*src_image, src_layout, *dst_image, dst_layout, regions, filter);
        keep_until_pool_reset(src_image);
        keep_until_pool_reset(dst_image);
    }

    void blit(const ImageHandle& src_image,
              const ImageHandle& dst_image,
              const vk::ArrayProxy<const vk::ImageBlit>& regions,
              const vk::Filter filter = vk::Filter::eLinear) {
        blit(src_image, src_image->get_current_layout(), dst_image, dst_image->get_current_layout(),
             regions, filter);
    }

    void clear(
        const ImageHandle& image,
        const vk::ImageLayout layout,
        const vk::ClearColorValue color = vk::ClearColorValue{0, 0, 0, 0},
        const vk::ArrayProxy<vk::ImageSubresourceRange>& ranges = merian::all_levels_and_layers()) {
        cmd.clearColorImage(*image, layout, color, ranges);
        keep_until_pool_reset(image);
    }

    void clear(
        const ImageHandle& image,
        const vk::ClearColorValue color = vk::ClearColorValue{0, 0, 0, 0},
        const vk::ArrayProxy<vk::ImageSubresourceRange>& ranges = merian::all_levels_and_layers()) {
        clear(image, image->get_current_layout(), color, ranges);
    }

    // ------------------------------------------------------------
    // IMAGE-BUFFER

    void copy(const ImageHandle& src_image,
              const vk::ImageLayout src_layout,
              const BufferHandle& dst_buffer,
              const vk::ArrayProxy<const vk::BufferImageCopy>& regions) {
        cmd.copyImageToBuffer(*src_image, src_layout, *dst_buffer, regions);
        keep_until_pool_reset(src_image);
        keep_until_pool_reset(dst_buffer);
    }

    void copy(const ImageHandle& src_image,
              const BufferHandle& dst_buffer,
              const vk::ArrayProxy<const vk::BufferImageCopy>& regions) {
        copy(src_image, src_image->get_current_layout(), dst_buffer, regions);
    }

    void copy(const BufferHandle& src_buffer,
              const ImageHandle& dst_image,
              const vk::ImageLayout dst_layout,
              const vk::ArrayProxy<const vk::BufferImageCopy>& regions) {
        cmd.copyBufferToImage(*src_buffer, *dst_image, dst_layout, regions);
        keep_until_pool_reset(src_buffer);
        keep_until_pool_reset(dst_image);
    }

    void copy(const BufferHandle& src_buffer,
              const ImageHandle& dst_image,
              const vk::ArrayProxy<const vk::BufferImageCopy>& regions) {
        copy(src_buffer, dst_image, dst_image->get_current_layout(), regions);
    }

    // ------------------------------------------------------------
    // PIPELINE

    void bind(const PipelineHandle& pipeline) {
        cmd.bindPipeline(pipeline->get_pipeline_bind_point(), *pipeline);
        keep_until_pool_reset(pipeline);
    }

    void bind_descriptor_set(const PipelineHandle& pipeline,
                             const DescriptorSetHandle& descriptor_set,
                             const uint32_t first_set = 0) {
        cmd.bindDescriptorSets(pipeline->get_pipeline_bind_point(), *pipeline->get_layout(),
                               first_set, 1, &**descriptor_set, 0, nullptr);
        keep_until_pool_reset(descriptor_set);
        keep_until_pool_reset(pipeline);
    }

    void push_descriptor_set(const PipelineHandle& pipeline,
                             const uint32_t set,
                             const std::vector<vk::WriteDescriptorSet>& writes) {
        cmd.pushDescriptorSetKHR(pipeline->get_pipeline_bind_point(), *pipeline->get_layout(), set,
                                 writes);
        keep_until_pool_reset(pipeline);
    }

    void push_descriptor_set(const PipelineHandle& pipeline,
                             const std::vector<vk::WriteDescriptorSet>& writes) {
        push_descriptor_set(pipeline, 0, writes);
    }

    template <typename... T>
    std::enable_if_t<(std::disjunction_v<std::is_same<vk::DescriptorBufferInfo, T>,
                                         std::is_same<vk::DescriptorImageInfo, T>> &&
                      ...)>
    push_descriptor_set(const PipelineHandle& pipeline, const uint32_t set, const T&... resources) {
        push_descriptor_set(pipeline, set, std::index_sequence_for<T...>{}, resources...);
    }

    template <typename... T>
    std::enable_if_t<(std::disjunction_v<std::is_same<vk::DescriptorBufferInfo, T>,
                                         std::is_same<vk::DescriptorImageInfo, T>> &&
                      ...)>
    push_descriptor_set(const PipelineHandle& pipeline, const T&... resources) {
        push_descriptor_set(pipeline, 0, std::index_sequence_for<T...>{}, resources...);
    }

    template <typename... T>
    std::enable_if_t<
        (std::disjunction_v<std::is_same<BufferHandle, T>, std::is_same<TextureHandle, T>> && ...)>
    push_descriptor_set(const PipelineHandle& pipeline, const uint32_t set, const T&... resources) {
        // need this recursive call else the Image and Buffer descriptor info is deallocated from
        // stack and we need the addresses.
        push_descriptor_set(pipeline, set, std::index_sequence_for<T...>{},
                            resources->get_descriptor_info()...);
    }

    template <typename... T>
    std::enable_if_t<
        (std::disjunction_v<std::is_same<BufferHandle, T>, std::is_same<TextureHandle, T>> && ...)>
    push_descriptor_set(const PipelineHandle& pipeline, const T&... resources) {
        // need this recursive call else the Image and Buffer descriptor info is deallocated from
        // stack and we need the addresses.
        push_descriptor_set(pipeline, 0, std::index_sequence_for<T...>{},
                            resources->get_descriptor_info()...);

        (keep_until_pool_reset(resources), ...);
    }

    template <typename T>
    void push_constant(const PipelineHandle& pipeline, const T& constant, const uint32_t id = 0) {
        push_constant(pipeline, reinterpret_cast<const void*>(&constant), id);
    }

    template <typename T>
    void push_constant(const PipelineHandle& pipeline, const T* constant, const uint32_t id = 0) {
        push_constant(pipeline, reinterpret_cast<const void*>(constant), id);
    }

    // The id that was returned by the pipeline layout builder.
    void push_constant(const PipelineHandle& pipeline, const void* values, const uint32_t id = 0) {
        auto range = pipeline->get_layout()->get_push_constant_range(id);
        push_constant(pipeline, range.stageFlags, range.offset, range.size, values);
    }

    void push_constant(const PipelineHandle& pipeline,
                       const vk::ShaderStageFlags flags,
                       const uint32_t offset,
                       const uint32_t size,
                       const void* values) {
        cmd.pushConstants(*pipeline->get_layout(), flags, offset, size, values);
        keep_until_pool_reset(pipeline);
    }

    void dispatch(const uint32_t group_count_x,
                  const uint32_t group_count_y = 1,
                  const uint32_t group_count_z = 1) {
        cmd.dispatch(group_count_x, group_count_y, group_count_z);
    }

    // computes the group count from the extent and local size.
    void dispatch(const vk::Extent3D& extent,
                  const uint32_t local_size_x,
                  const uint32_t local_size_y = 1,
                  const uint32_t local_size_z = 1) {
        dispatch((extent.width + local_size_x - 1) / local_size_x,
                 (extent.height + local_size_y - 1) / local_size_y,
                 (extent.depth + local_size_z - 1) / local_size_z);
    }

    // computes the group count from the extent and local size.
    void
    dispatch(const vk::Extent2D& extent, const uint32_t local_size_x, const uint32_t local_size_y) {
        dispatch((extent.width + local_size_x - 1) / local_size_x,
                 (extent.height + local_size_y - 1) / local_size_y, 1);
    }

    // ------------------------------------------------------------
    // MISC

    void set_event(const EventHandle& event, const vk::PipelineStageFlags stage_mask) {
        cmd.setEvent(*event, stage_mask);
        keep_until_pool_reset(event);
    }

    void set_event(const EventHandle& event, const vk::DependencyInfo& dep_info) {
        cmd.setEvent2(*event, dep_info);
        keep_until_pool_reset(event);
    }

    void set_event(const EventHandle& event,
                   const vk::ArrayProxyNoTemporaries<const vk::MemoryBarrier2>& memory_barriers) {
        cmd.setEvent2(*event, vk::DependencyInfo{{}, memory_barriers});
        keep_until_pool_reset(event);
    }

    void set_event(
        const EventHandle& event,
        const vk::ArrayProxyNoTemporaries<const vk::BufferMemoryBarrier2>& buffer_memory_barriers) {
        cmd.setEvent2(*event, vk::DependencyInfo{{}, {}, buffer_memory_barriers});
        keep_until_pool_reset(event);
    }

    void set_event(
        const EventHandle& event,
        const vk::ArrayProxyNoTemporaries<const vk::ImageMemoryBarrier2>& image_memory_barriers) {
        cmd.setEvent2(*event, vk::DependencyInfo{{}, {}, {}, image_memory_barriers});
        keep_until_pool_reset(event);
    }

    // ------------------------------------------------------------
    // ACCELERATION STRUCTURE

    void copy_acceleration_structure(const HWAccelerationStructureHandle& src,
                                     const HWAccelerationStructureHandle& dst,
                                     const vk::CopyAccelerationStructureModeKHR mode =
                                         vk::CopyAccelerationStructureModeKHR::eClone) {
        cmd.copyAccelerationStructureKHR(vk::CopyAccelerationStructureInfoKHR{*src, *dst, mode});
        keep_until_pool_reset(src);
        keep_until_pool_reset(dst);
    }

    // ------------------------------------------------------------
    // QUERY POOL

    template <vk::QueryType QUERY_TYPE>
    void reset(const QueryPoolHandle<QUERY_TYPE> query_pool,
               const uint32_t first_query,
               const uint32_t query_count) {
        cmd.resetQueryPool(*query_pool, first_query, query_count);
        keep_until_pool_reset(query_pool);
    }

    template <vk::QueryType QUERY_TYPE> void reset(const QueryPoolHandle<QUERY_TYPE> query_pool) {
        cmd.resetQueryPool(*query_pool, 0, query_pool->get_query_count());
        keep_until_pool_reset(query_pool);
    }

    void write_timestamp(
        const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool,
        const uint32_t query = 0,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands) {
        assert(query < query_pool->get_query_count());
        cmd.writeTimestamp(pipeline_stage, *query_pool, query);
        keep_until_pool_reset(query_pool);
    }

    void write_timestamp(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool,
                         const uint32_t query = 0,
                         const vk::PipelineStageFlagBits2 pipeline_stage =
                             vk::PipelineStageFlagBits2::eAllCommands) {
        assert(query < query_pool->get_query_count());
        cmd.writeTimestamp2(pipeline_stage, *query_pool, query);
        keep_until_pool_reset(query_pool);
    }

    void write_acceleration_structures_properties(
        const QueryPoolHandle<vk::QueryType::eAccelerationStructureCompactedSizeKHR>& query_pool,
        const vk::ArrayProxy<const HWAccelerationStructureHandle> ass,
        uint32_t first_query = 0) {

        std::vector<vk::AccelerationStructureKHR> acc_structures(ass.size());
        std::transform(ass.begin(), ass.end(), acc_structures.begin(), [&](auto& as) {
            keep_until_pool_reset(as);
            return as->get_acceleration_structure();
        });

        cmd.writeAccelerationStructuresPropertiesKHR(
            acc_structures, vk::QueryType::eAccelerationStructureCompactedSizeKHR, *query_pool,
            first_query);

        keep_until_pool_reset(query_pool);
    }

    // ------------------------------------------------------------
    // BARRIERS

    void barrier(vk::PipelineStageFlags src_stage_mask,
                 vk::PipelineStageFlags dst_stage_mask,
                 const vk::ArrayProxy<const vk::MemoryBarrier>& memory_barriers) {
        cmd.pipelineBarrier(src_stage_mask, dst_stage_mask, {}, memory_barriers, {}, {});
    }

    void barrier(vk::PipelineStageFlags src_stage_mask,
                 vk::PipelineStageFlags dst_stage_mask,
                 const vk::ArrayProxy<const vk::BufferMemoryBarrier>& buffer_memory_barriers) {
        cmd.pipelineBarrier(src_stage_mask, dst_stage_mask, {}, {}, buffer_memory_barriers, {});
    }

    void barrier(vk::PipelineStageFlags src_stage_mask,
                 vk::PipelineStageFlags dst_stage_mask,
                 const vk::ArrayProxy<const vk::ImageMemoryBarrier>& image_memory_barriers) {
        cmd.pipelineBarrier(src_stage_mask, dst_stage_mask, {}, {}, {}, image_memory_barriers);
    }

    void barrier(const vk::DependencyInfo& dep_info) {
        cmd.pipelineBarrier2(dep_info);
    }

    void barrier(const vk::ArrayProxy<const vk::MemoryBarrier2>& memory_barriers,
                 const vk::ArrayProxy<const vk::BufferMemoryBarrier2>& buffer_memory_barriers = {},
                 const vk::ArrayProxy<const vk::ImageMemoryBarrier2>& image_memory_barriers = {}) {
        cmd.pipelineBarrier2(
            vk::DependencyInfo{{}, memory_barriers, buffer_memory_barriers, image_memory_barriers});
    }

    void barrier(const vk::ArrayProxy<const vk::BufferMemoryBarrier2>& buffer_memory_barriers) {
        cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, buffer_memory_barriers});
    }

    void barrier(const vk::ArrayProxy<const vk::ImageMemoryBarrier2>& image_memory_barriers) {
        cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, image_memory_barriers});
    }

    // ------------------------------------------------------------

  private:
    static vk::WriteDescriptorSet make_descriptor_write(const vk::DescriptorBufferInfo& buffer_info,
                                                        const DescriptorSetLayoutHandle& set_layout,
                                                        const uint32_t binding) {

        return vk::WriteDescriptorSet{
            {}, binding, 0, 1, set_layout->get_type_for_binding(binding), nullptr, &buffer_info,
        };
    }

    static vk::WriteDescriptorSet make_descriptor_write(const vk::DescriptorImageInfo& image_info,
                                                        const DescriptorSetLayoutHandle& set_layout,
                                                        const uint32_t binding) {
        return vk::WriteDescriptorSet{
            {}, binding, 0, 1, set_layout->get_type_for_binding(binding), &image_info, nullptr,
        };
    }

    template <typename... T, std::size_t... Is>
    void push_descriptor_set(const PipelineHandle& pipeline,
                             const uint32_t set,
                             const std::index_sequence<Is...> /*unused*/,
                             const T&... resources) {
        const DescriptorSetLayoutHandle set_layout =
            pipeline->get_layout()->get_descriptor_set_layout(set);

        const std::vector<vk::WriteDescriptorSet> writes = {
            make_descriptor_write(resources, set_layout, Is)...};

        push_descriptor_set(pipeline, set, writes);
    }

  private:
    const CommandPoolHandle pool;

    vk::CommandBuffer cmd;

  public:
    static CommandBufferHandle
    create(const CommandPoolHandle& pool,
           const vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) {
        return std::make_shared<CommandBuffer>(pool, level);
    }
};

} // namespace merian
