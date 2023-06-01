#pragma once

#include "merian/vk/descriptors/descriptor_set_layout.hpp"

#include <map>
#include <spdlog/spdlog.h>
#include <tuple>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

/**
 * @brief      Builds DescriptorSetLayouts, corresponding DescriptorPools, allocates DescriptorSets and performs error
 * checking.
 *
 * A Descriptor can be seen as a pointer to a resource.
 *
 * A DescriptorSetLayout describes the resources of a shader and their bindings (the order in the shader)
 * and can be seen as the function signature (excluding push constants) of a shader.
 *
 * Example:
 * DescriptorSetLayout                  DescriptorSet
 *
 * 0, Sampler[2]            ->          <descriptor addr>
 *                          ->          <descriptor addr>
 * 1, StorageBuffer         ->          <descriptor addr>
 *
 * Since Descriptors use some memory, they need to be allocated from DescriptorPools.
 * A DescriptorPool can allocate space for a certain amount of Descriptors and a max amount of DescriptorSets.
 * The Descriptors can be arbitraryly be used for example to create to equal DescriptorSets or one that uses more
 * Descriptors as the other, as long as no more that the maximum number of Descriptors is not passed.
 *
 * This builder can create a DescriptorPool that contains enough Descriptors for a certain number of DescriptorSets
 * following this Layout. If you want to use multiple different layouts and use one DescriptorPool you may want to look
 * into DescriptorPoolBuilder.
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
    // Determines the pool sizes to generate set_count DescriptorSets with these bindings.
    // E.g. set_count = 1 means with these sizes exactly once DescriptorSet with all bindings can be created (or two
    // with half of the bindings).
    static std::vector<vk::DescriptorPoolSize>
    make_pool_sizes_from_bindings(const std::vector<vk::DescriptorSetLayoutBinding>& bindings, uint32_t set_count = 1) {

        // We make one entry for each "type" of binding.
        std::vector<vk::DescriptorPoolSize> sizes;
        std::map<vk::DescriptorType, uint32_t> type_to_index;

        for (auto& binding : bindings) {
            if (!type_to_index.contains(binding.descriptorType)) {
                vk::DescriptorPoolSize size_for_type{binding.descriptorType, binding.descriptorCount * set_count};
                type_to_index[binding.descriptorType] = sizes.size();
                sizes.push_back(size_for_type);
            } else {
                sizes[type_to_index[binding.descriptorType]].descriptorCount += binding.descriptorCount * set_count;
            }
        }

        return sizes;
    }

    // Creates a vk::DescriptorPool that has enough Descriptors such that set_count DescriptorSets with these bindings
    // can be created. By default the pools max sets property is set to set_count, you can override that if this pool
    // should be used for different layouts for example.
    static vk::DescriptorPool descriptor_pool_for_bindings(const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                                                           const vk::Device& device,
                                                           const uint32_t set_count = 1,
                                                           const vk::DescriptorPoolCreateFlags flags = {},
                                                           const std::optional<uint32_t> max_sets = std::nullopt) {
        std::vector<vk::DescriptorPoolSize> pool_sizes = make_pool_sizes_from_bindings(bindings, set_count);
        vk::DescriptorPoolCreateInfo info{flags, max_sets.value_or(set_count), pool_sizes};
        return device.createDescriptorPool(info);
    }

  public:
    DescriptorSetLayoutBuilder(std::vector<vk::DescriptorSetLayoutBinding> bindings = {}) {
        add_binding(bindings);
    }

    // --------------------------------------------------------------------------------------------------------------------

    DescriptorSetLayoutBuilder&
    add_binding_storage_buffer(vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
                               uint32_t descriptor_count = 1,
                               const vk::Sampler* sampler = nullptr,
                               std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eStorageBuffer, descriptor_count, sampler, binding);
        return *this;
    }

    DescriptorSetLayoutBuilder&
    add_binding_sampler(vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
                        uint32_t descriptor_count = 1,
                        const vk::Sampler* sampler = nullptr,
                        std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eSampler, descriptor_count, sampler, binding);
        return *this;
    }

    DescriptorSetLayoutBuilder&
    add_binding_acceleration_structure(vk::ShaderStageFlags stage_flags = vk::ShaderStageFlagBits::eCompute,
                                       uint32_t descriptor_count = 1,
                                       const vk::Sampler* sampler = nullptr,
                                       std::optional<uint32_t> binding = std::nullopt) {
        add_binding(stage_flags, vk::DescriptorType::eAccelerationStructureKHR, descriptor_count, sampler, binding);
        return *this;
    }

    // --------------------------------------------------------------------------------------------------------------------

    /**
     * @brief      Adds a binding to the descriptor set.
     *
     *
     * @param[in]  binding              The binding point - this is the binding index that is referenced in the layout.
     * // if no value is suppied for `binding`, then smallest positive integer without binding is used in the shader
     * @param[in]  descriptor_type_     The descriptor type
     * @param[in]  descriptor_count     The descriptor count
     * @param[in]  stage_flags          The stage slags
     * @param[in]  pImmutableSamplers_  Pointer to a sampler, used for textures
     */
    DescriptorSetLayoutBuilder& add_binding(vk::ShaderStageFlags stage_flags,
                                            vk::DescriptorType descriptor_type = vk::DescriptorType::eSampler,
                                            uint32_t descriptor_count = 1,
                                            const vk::Sampler* sampler = nullptr,
                                            std::optional<uint32_t> binding = std::nullopt) {
        vk::DescriptorSetLayoutBinding layout_binding{binding.value_or(next_free_binding()), descriptor_type,
                                                      descriptor_count, stage_flags, sampler};
        add_binding(layout_binding);
        return *this;
    }

    DescriptorSetLayoutBuilder& add_binding(std::vector<vk::DescriptorSetLayoutBinding> bindings) {
        for (auto& binding : bindings) {
            add_binding(binding);
        }
        return *this;
    }

    DescriptorSetLayoutBuilder& add_binding(vk::DescriptorSetLayoutBinding binding) {
#ifdef DEBUG
        if (binding_indices.contains(binding.binding)) {
            SPDLOG_WARN("builder already contains a binding with binding point {}", binding.binding);
        }
        binding_indices[binding.binding] = bindings.size();
#endif
        bindings.push_back(binding);
        return *this;
    }

    // --------------------------------------------------------------------------------------------------------------------

    // Requires that there is a binding from 0 to num_bindings-1.
    // Return a shared ptr since many descriptor sets may have a reference on this.
    std::shared_ptr<DescriptorSetLayout> build_layout(vk::Device& device, vk::DescriptorSetLayoutCreateFlags flags = {}) {
        vk::DescriptorSetLayoutCreateInfo info{flags, bindings};

        std::vector<vk::DescriptorType> types(bindings.size());
        for (uint32_t i = 0; i < bindings.size(); i++) {
            if (!binding_indices.contains(i)) {
                throw std::runtime_error{
                    fmt::format("this builder has {} bindings, but binding {} is missing!", bindings.size(), i)};
            } else {
                types.push_back(bindings[binding_indices[i]].descriptorType);
            }
        }

        return make_shared<DescriptorSetLayout>(device.createDescriptorSetLayout(info), std::move(types));
    }

    vk::DescriptorPool build_pool(const vk::Device& device,
                                  const uint32_t set_count = 1,
                                  const vk::DescriptorPoolCreateFlags flags = {},
                                  const std::optional<uint32_t> max_sets = std::nullopt) {
        return descriptor_pool_for_bindings(bindings, device, set_count, flags, max_sets);
    }

    // --------------------------------------------------------------------------------------------------------------------

  private:
    uint32_t next_free_binding() {
        for (uint32_t i = 0; i <= bindings.size(); i++) {
            if (!binding_indices.contains(i)) {
                return i;
            }
        }
        throw std::runtime_error{"this should not happen"};
    }

  private:
    std::map<uint32_t, uint32_t> binding_indices;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
};

} // namespace merian
