#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/shader_object_layout.hpp"
#include "merian/shader/slang_utils.hpp"
#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

/**
 * @brief Runtime shader parameter container. Behavior follows the layout kind:
 *
 * ParameterBlock objects own the descriptor set machinery, the implicit uniform buffer and an
 * element object. ConstantBuffer objects own their uniform buffer and an element object. Struct
 * objects own no Vulkan resources: uniform writes go to the owning container's staging buffer,
 * descriptor writes are recorded relative to the struct layout and replayed into the descriptor
 * set of every ParameterBlock the struct is attached to.
 */
class ShaderObject : public std::enable_shared_from_this<ShaderObject> {
  public:
    ShaderObject(const ShaderObjectLayoutHandle& object_layout,
                 const ResourceAllocatorHandle& allocator);

    // Containers return a cursor into their element.
    ShaderCursor get_cursor();

    // The element object of a container, created lazily.
    const ShaderObjectHandle& get_element();

    // ---------------------------------------------------------------
    // Sub-objects (struct kind; containers forward to their element)

    ShaderObjectHandle create_subobject(const std::string& field_name);

    void set_subobject(const std::string& field_name, const ShaderObjectHandle& object);

    // Writes the ConstantBuffer's uniform-buffer descriptor into every attached ParameterBlock
    // at set time (not bind time) so frames with no state changes issue no descriptor writes.
    void set_subobject(uint32_t subobject_range_index, const ShaderObjectHandle& object);

    const ShaderObjectHandle& get_subobject(uint32_t subobject_range_index);

    uint32_t get_subobject_count() const {
        return object_layout->get_subobject_range_count();
    }

    // ---------------------------------------------------------------
    // Binding (ParameterBlock kind)

    // True if this block or any nested ParameterBlock records a uniform upload on its next
    // bind. Lets callers elide transfer barriers on clean frames.
    bool has_pending_uploads();

    // The obj_allocator supplies the descriptor container; not stored. Nested ParameterBlock
    // sub-objects are NOT bound here — see SlangProgramEntryPoint. The caller must barrier
    // uniform buffers when has_pending_uploads() (one barrier for the whole bind walk).
    void bind_as_parameter_block(const CommandBufferHandle& cmd,
                                 const PipelineHandle& pipeline,
                                 uint32_t set_index,
                                 const ShaderObjectAllocatorHandle& obj_allocator);

    // ---------------------------------------------------------------
    // Writes (struct kind, called through ShaderCursor)

    void write(const ShaderOffset& offset,
               const ImageViewHandle& image,
               const std::optional<vk::ImageLayout> access_layout = std::nullopt);
    void write(const ShaderOffset& offset, const BufferHandle& buffer);
    void write(const ShaderOffset& offset,
               const TextureHandle& texture,
               const std::optional<vk::ImageLayout> access_layout = std::nullopt);
    void write(const ShaderOffset& offset, const SamplerHandle& sampler);
    void write(const ShaderOffset& offset, const AccelerationStructureHandle& as);
    void write(const ShaderOffset& offset, const void* data, std::size_t size);

    // ---------------------------------------------------------------

    slang::TypeLayoutReflection* get_type_layout() const {
        return object_layout->get_type_layout();
    }

    const ShaderObjectLayoutHandle& get_object_layout() const {
        return object_layout;
    }

    const ResourceAllocatorHandle& get_allocator() const {
        return allocator;
    }

  private:
    struct ImageSlot {
        ImageViewHandle image;
        std::optional<vk::ImageLayout> access_layout;
    };
    struct TextureSlot {
        TextureHandle texture;
        std::optional<vk::ImageLayout> access_layout;
    };
    // Recorded descriptor write, replayed on attach.
    using ResourceSlot = std::variant<std::monostate,
                                      ImageSlot,
                                      BufferHandle,
                                      TextureSlot,
                                      SamplerHandle,
                                      AccelerationStructureHandle>;

    // A ParameterBlock whose descriptor set receives this struct's descriptor writes,
    // shifted by binding_base.
    struct DescriptorTarget {
        std::weak_ptr<ShaderObject> parameter_block;
        uint32_t binding_base;
    };

    // ---- struct kind

    void attach(const ShaderObjectHandle& parameter_block, uint32_t binding_base);
    void detach(const ShaderObject* parameter_block, uint32_t binding_base);
    void replay_to(ShaderObject& parameter_block, uint32_t binding_base);
    void attach_constant_buffer(const ShaderObjectHandle& constant_buffer,
                                const ShaderObjectLayout::SubobjectRangeInfo& range,
                                const ShaderObjectHandle& parameter_block,
                                uint32_t binding_base);
    void
    for_each_descriptor_target(const std::function<void(ShaderObject&, uint32_t binding_base)>& fn);
    void apply_slot(const ResourceSlot& slot,
                    ShaderObject& parameter_block,
                    uint32_t binding,
                    uint32_t array_index);

    // ---- container kinds

    void write_uniform(std::size_t byte_offset, const void* data, std::size_t size);
    // Flags every ParameterBlock that must upload this container's staging on its next bind.
    void mark_uploads_pending();
    void upload_uniform_buffers(const CommandBufferHandle& cmd);

    // ---- ParameterBlock kind

    // Applies a write to the descriptor storage and all live descriptor sets.
    void for_each_descriptor_container(const std::function<void(DescriptorContainer&)>& fn);

  private:
    ShaderObjectLayoutHandle object_layout;
    ResourceAllocatorHandle allocator;

    // struct kind
    std::weak_ptr<ShaderObject> owner_container;
    std::vector<DescriptorTarget> descriptor_targets;
    std::vector<ResourceSlot> slots;
    std::vector<ShaderObjectHandle> subobjects;

    // container kinds
    ShaderObjectHandle element = nullptr;
    BufferHandle uniform_buffer = nullptr;
    std::vector<uint8_t> uniform_staging;
    bool uniform_dirty = false;

    // ParameterBlock kind
    DescriptorStorageHandle descriptors = nullptr;
    std::vector<std::weak_ptr<DescriptorContainer>> registered_sets;
    bool uploads_pending = false;

    friend class ShaderCursor;
    friend std::string format_as(const ShaderObject& shader_object, const std::string& indent);
};

std::string format_as(const ShaderObject& shader_object, const std::string& indent = "");

} // namespace merian
