#pragma once

#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <map>
#include <optional>
#include <spdlog/spdlog.h>
#include <tuple>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

/**
 * @brief      Builds DescriptorSetLayouts, corresponding DescriptorPools, allocates DescriptorSets
 * and performs error checking.
 *
 * A Descriptor can be seen as a pointer to a resource.
 *
 * A DescriptorSetLayout describes the resources of a shader and their bindings (the order in the
 * shader) and can be seen as the function signature (excluding push constants) of a shader.
 *
 * Example:
 * DescriptorSetLayout                  DescriptorSet
 *
 * 0, Sampler[2]            ->          <descriptor addr>
 *                          ->          <descriptor addr>
 * 1, StorageBuffer         ->          <descriptor addr>
 *
 * Since Descriptors use some memory, they need to be allocated from DescriptorPools.
 * A DescriptorPool can allocate space for a certain amount of Descriptors and a max amount of
 * DescriptorSets. The Descriptors can be arbitraryly be used for example to create to equal
 * DescriptorSets or one that uses more Descriptors as the other, as long as no more that the
 * maximum number of Descriptors is not passed.
 *
 * This builder can create a DescriptorPool that contains enough Descriptors for a certain number of
 * DescriptorSets following this Layout. If you want to use multiple different layouts and use one
 * DescriptorPool you may want to look into DescriptorPoolBuilder.
 *
 * E.g for a shader with ping pong buffers in and out:
 * In the shader
 * layout (binding = 0) buffer in {float v_in[];};
 * layout (binding = 1) buffer out {float v_out[];};
 *
 * auto builder = DescriptorSetLayoutBuilder()
 *      .add_binding_storage_buffer()
 *      .add_binding_storage_buffer();
 *
 * auto layout = builder.build_layout(context);
 * auto pool = builder.build_pool(context, 2); // 2 -> one set for ping one for pong
 * auto sets = allocate_descriptor_sets(context, pool, layout, 2); -> one for ping one for pong
 * ...
 * destroy sets, pool, layout
 */
class DescriptorSetLayoutBuilder {

  public:
    DescriptorSetLayoutBuilder(const std::vector<vk::DescriptorSetLayoutBinding>& bindings = {}) {
        add_binding(bindings);
    }

    // --------------------------------------------------------------------------------------------------------------------

    DescriptorSetLayoutBuilder&
    add_binding_storage_buffer(vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
                               uint32_t descriptor_count = 1,
                               std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eStorageBuffer, descriptor_count, nullptr,
                    binding);
        return *this;
    }

    DescriptorSetLayoutBuilder&
    add_binding_uniform_buffer(vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
                               uint32_t descriptor_count = 1,
                               std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eUniformBuffer, descriptor_count, nullptr,
                    binding);
        return *this;
    }

    DescriptorSetLayoutBuilder&
    add_binding_storage_image(vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
                              uint32_t descriptor_count = 1,
                              std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eStorageImage, descriptor_count, nullptr,
                    binding);
        return *this;
    }

    // immutable_sampler can be used to initialize a set of immutable samplers. Immutable samplers
    // are permanently bound into the set layout and must not be changed; updating a
    // VK_DESCRIPTOR_TYPE_SAMPLER descriptor with immutable samplers is not allowed.
    DescriptorSetLayoutBuilder&
    add_binding_sampler(vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
                        uint32_t descriptor_count = 1,
                        const vk::Sampler* immutable_sampler = nullptr,
                        std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eSampler, descriptor_count, immutable_sampler,
                    binding);
        return *this;
    }

    // pImmutableSamplers can be used to initialize a set of immutable samplers. Immutable samplers
    // are permanently bound into the set layout and must not be changed; updates to a
    // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER descriptor with immutable samplers does not modify
    // the samplers (the image views are updated, but the sampler updates are ignored).
    DescriptorSetLayoutBuilder& add_binding_combined_sampler(
        vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
        uint32_t descriptor_count = 1,
        const vk::Sampler* immutable_sampler = nullptr,
        std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eCombinedImageSampler, descriptor_count,
                    immutable_sampler, binding);
        return *this;
    }

    DescriptorSetLayoutBuilder& add_binding_acceleration_structure(
        vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
        uint32_t descriptor_count = 1,
        const vk::Sampler* immutable_sampler = nullptr,
        std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eAccelerationStructureKHR, descriptor_count,
                    immutable_sampler, binding);
        return *this;
    }

    // --------------------------------------------------------------------------------------------------------------------

    /**
     * @brief      Adds a binding to the descriptor set.
     *
     *
     * @param[in]  binding              The binding point - this is the binding index that is
     * referenced in the layout.
     * // if no value is suppied for `binding`, then smallest positive integer without binding is
     * used in the shader
     * @param[in]  descriptor_type_     The descriptor type
     * @param[in]  descriptor_count     The descriptor count
     * @param[in]  stage_flags          The stage slags
     * @param[in]  pImmutableSamplers_  See vulkan spec
     */
    DescriptorSetLayoutBuilder&
    add_binding(vk::ShaderStageFlags stage_flags,
                vk::DescriptorType descriptor_type = vk::DescriptorType::eSampler,
                uint32_t descriptor_count = 1,
                const vk::Sampler* immutable_sampler = nullptr,
                std::optional<uint32_t> binding = std::nullopt) {
        vk::DescriptorSetLayoutBinding layout_binding{binding.value_or(next_free_binding()),
                                                      descriptor_type, descriptor_count,
                                                      stage_flags, immutable_sampler};
        add_binding(layout_binding);
        return *this;
    }

    DescriptorSetLayoutBuilder&
    add_binding(const std::vector<vk::DescriptorSetLayoutBinding>& bindings) {
        for (const auto& binding : bindings) {
            add_binding(binding);
        }
        return *this;
    }

    DescriptorSetLayoutBuilder& add_binding(vk::DescriptorSetLayoutBinding binding) {
#ifndef NDEBUG
        if (bindings.contains(binding.binding)) {
            SPDLOG_WARN("builder already contains a binding with binding point {}",
                        binding.binding);
        }
#endif
        bindings[binding.binding] = binding;
        return *this;
    }

    // --------------------------------------------------------------------------------------------------------------------

    // Requires that there is a binding from 0 to num_bindings-1.
    // Return a shared ptr since many descriptor sets may have a reference on this.
    DescriptorSetLayoutHandle build_layout(const ContextHandle& context,
                                           const vk::DescriptorSetLayoutCreateFlags flags = {}) {

        std::vector<vk::DescriptorSetLayoutBinding> sorted_bindings(bindings.size());

        for (uint32_t i = 0; i < bindings.size(); i++) {
            assert(bindings.contains(i));
            sorted_bindings[i] = bindings[i];
        }

        return make_shared<DescriptorSetLayout>(context, std::move(sorted_bindings), flags);
    }

    DescriptorSetLayoutHandle
    build_push_descriptor_layout(const ContextHandle& context,
                                 const vk::DescriptorSetLayoutCreateFlags flags = {}) {
        return build_layout(context,
                            flags | vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR);
    }

    // --------------------------------------------------------------------------------------------------------------------

  private:
    uint32_t next_free_binding() {
        for (uint32_t i = 0; i <= bindings.size(); i++) {
            if (!bindings.contains(i)) {
                return i;
            }
        }
        throw std::runtime_error{"this should not happen"};
        // since we check one index more than size()
    }

  private:
    std::map<uint32_t, vk::DescriptorSetLayoutBinding> bindings;
};

} // namespace merian
