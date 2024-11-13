#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include <spdlog/spdlog.h>

namespace merian {

class Pipeline : public std::enable_shared_from_this<Pipeline> {

  public:
    Pipeline(const ContextHandle& context, const std::shared_ptr<PipelineLayout>& pipeline_layout)
        : context(context), pipeline_layout(pipeline_layout) {}

    virtual ~Pipeline(){};

    // ---------------------------------------------------------------------------

    operator const vk::Pipeline&() const {
        return pipeline;
    }

    const vk::Pipeline& get_pipeline() {
        return pipeline;
    }

    const std::shared_ptr<PipelineLayout>& get_layout() {
        return pipeline_layout;
    }

    void bind(const vk::CommandBuffer& cmd) {
        cmd.bindPipeline(get_pipeline_bind_point(), pipeline);
    }

    void bind_descriptor_set(const vk::CommandBuffer& cmd,
                             const std::shared_ptr<DescriptorSet>& descriptor_set,
                             const uint32_t first_set = 0) {
        cmd.bindDescriptorSets(get_pipeline_bind_point(), *pipeline_layout, first_set, 1,
                               &**descriptor_set, 0, nullptr);
    }

    void push_descriptor_set(const vk::CommandBuffer& cmd,
                             const uint32_t set,
                             const std::vector<vk::WriteDescriptorSet>& writes) {
        cmd.pushDescriptorSetKHR(get_pipeline_bind_point(), *pipeline_layout, set, writes);
    }

    // shortcut to push the descriptor set 0.
    void push_descriptor_set(const vk::CommandBuffer& cmd,
                             const std::vector<vk::WriteDescriptorSet>& writes) {
        push_descriptor_set(cmd, 0, writes);
    }

  private:
    vk::WriteDescriptorSet make_descriptor_write(const vk::DescriptorBufferInfo& buffer_info,
                                                 const uint32_t set,
                                                 const uint32_t binding) {

        return vk::WriteDescriptorSet{
            {},
            binding,
            0,
            1,
            pipeline_layout->get_descriptor_set_layout(set)->get_type_for_binding(binding),
            nullptr,
            &buffer_info,
        };
    }

    vk::WriteDescriptorSet make_descriptor_write(const vk::DescriptorImageInfo& image_info,
                                                 const uint32_t set,
                                                 const uint32_t binding) {
        return vk::WriteDescriptorSet{
            {},
            binding,
            0,
            1,
            pipeline_layout->get_descriptor_set_layout(set)->get_type_for_binding(binding),
            &image_info,
            nullptr,
        };
    }

    template <typename... T, std::size_t... Is>
    void push_descriptor_set(const vk::CommandBuffer& cmd,
                             const uint32_t set,
                             const std::index_sequence<Is...> /*unused*/,
                             const T&... resources) {
        const std::vector<vk::WriteDescriptorSet> writes = {
            make_descriptor_write(resources, set, Is)...};
        push_descriptor_set(cmd, set, writes);
    }

  public:
    template <typename... T>
    std::enable_if_t<(std::disjunction_v<std::is_same<vk::DescriptorBufferInfo, T>,
                                         std::is_same<vk::DescriptorImageInfo, T>> &&
                      ...)>
    push_descriptor_set(const vk::CommandBuffer& cmd, const uint32_t set, const T&... resources) {
        push_descriptor_set(cmd, set, std::index_sequence_for<T...>{}, resources...);
    }

    template <typename... T>
    std::enable_if_t<(std::disjunction_v<std::is_same<vk::DescriptorBufferInfo, T>,
                                         std::is_same<vk::DescriptorImageInfo, T>> &&
                      ...)>
    push_descriptor_set(const vk::CommandBuffer& cmd, const T&... resources) {
        push_descriptor_set(cmd, 0, std::index_sequence_for<T...>{}, resources...);
    }

    template <typename... T>
    std::enable_if_t<
        (std::disjunction_v<std::is_same<BufferHandle, T>, std::is_same<TextureHandle, T>> && ...)>
    push_descriptor_set(const vk::CommandBuffer& cmd, const uint32_t set, const T&... resources) {
        // need this recursive call else the Image and Buffer descriptor info is deallocated from
        // stack and we need the addresses.
        push_descriptor_set(cmd, set, std::index_sequence_for<T...>{},
                            resources->get_descriptor_info()...);
    }

    template <typename... T>
    std::enable_if_t<
        (std::disjunction_v<std::is_same<BufferHandle, T>, std::is_same<TextureHandle, T>> && ...)>
    push_descriptor_set(const vk::CommandBuffer& cmd, const T&... resources) {
        // need this recursive call else the Image and Buffer descriptor info is deallocated from
        // stack and we need the addresses.
        push_descriptor_set(cmd, 0, std::index_sequence_for<T...>{},
                            resources->get_descriptor_info()...);
    }

    // ---------------------------------------------------------------------------

    virtual vk::PipelineBindPoint get_pipeline_bind_point() const = 0;

    template <typename T>
    void push_constant(const vk::CommandBuffer& cmd, const T& constant, const uint32_t id = 0) {
        push_constant(cmd, reinterpret_cast<const void*>(&constant), id);
    }

    template <typename T>
    void push_constant(const vk::CommandBuffer& cmd, const T* constant, const uint32_t id = 0) {
        push_constant(cmd, reinterpret_cast<const void*>(constant), id);
    }

    // The id that was returned by the pipeline layout builder.
    void push_constant(const vk::CommandBuffer& cmd, const void* values, const uint32_t id = 0) {
        auto range = pipeline_layout->get_push_constant_range(id);
        push_constant(cmd, range.stageFlags, range.offset, range.size, values);
    }

    virtual void push_constant(const vk::CommandBuffer& cmd,
                               const vk::ShaderStageFlags flags,
                               const uint32_t offset,
                               const uint32_t size,
                               const void* values) {
        cmd.pushConstants(*pipeline_layout, flags, offset, size, values);
    }

  protected:
    const ContextHandle context;
    const std::shared_ptr<PipelineLayout> pipeline_layout;
    vk::Pipeline pipeline;
    vk::PipelineBindPoint bind_point;
};

using PipelineHandle = std::shared_ptr<Pipeline>;

} // namespace merian
