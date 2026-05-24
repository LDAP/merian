#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/shader_object_layout.hpp"
#include "merian/shader/slang_utils.hpp"
#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "slang.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

class ShaderObject : public std::enable_shared_from_this<ShaderObject> {
  public:
    ShaderObject(const ShaderObjectLayoutHandle& object_layout,
                 const ResourceAllocatorHandle& allocator);

    ~ShaderObject() = default;

    // The obj_allocator supplies the descriptor container; not stored. ParameterBlock sub-objects
    // are NOT bound here — see SlangProgramEntryPoint::bind_entry_point_parameter for those.
    // The caller must barrier uniform buffers (one barrier for the whole bind walk).
    void bind_as_parameter_block(const CommandBufferHandle& cmd,
                                 const PipelineHandle& pipeline,
                                 uint32_t set_index,
                                 const ShaderObjectAllocatorHandle& obj_allocator);

    ShaderCursor get_cursor();

    ShaderObjectHandle create_subobject(const std::string& field_name);

    void set_subobject(const std::string& field_name, const ShaderObjectHandle& object);

    const ShaderObjectHandle& get_subobject(uint32_t subobject_range_index) const {
        assert(subobject_range_index < subobjects.size());
        return subobjects[subobject_range_index];
    }

    // Writes the CB's UBO into the owning PB's storage at set time (not bind time) so frames with
    // no state changes issue no descriptor writes.
    void set_subobject(uint32_t subobject_range_index, const ShaderObjectHandle& object);

    uint32_t get_subobject_count() const {
        return static_cast<uint32_t>(subobjects.size());
    }

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
    void upload_constant_buffer_tree(ShaderObject* cb_obj, const CommandBufferHandle& cmd);

    void for_each_registered_set(const std::function<void(DescriptorContainer&)>& fn);

  private:
    ShaderObjectLayoutHandle object_layout;
    ResourceAllocatorHandle allocator;

    DescriptorStorageHandle descriptors = nullptr;

    std::vector<std::weak_ptr<DescriptorContainer>> registered_sets;

    BufferHandle ordinary_data_buffer;
    std::vector<uint8_t> ordinary_data_staging;
    bool ordinary_data_dirty = false;

    std::vector<ShaderObjectHandle> subobjects;

    struct OwnerConstantBufferBindings {
        std::weak_ptr<ShaderObject> pb;
        uint32_t element_binding;
    };
    // When this object is a ConstantBuffer field, we write into the owning PB's descriptor storage
    // (not the immediate parent's). Multiple owners are possible if the same CB is set in several
    // places.
    std::vector<OwnerConstantBufferBindings> cb_owners;

    friend class ShaderCursor;
    friend class SlangProgramEntryPoint;
    friend std::string format_as(const ShaderObject& shader_object, const std::string& indent);
};

std::string format_as(const ShaderObject& shader_object, const std::string& indent = "");

} // namespace merian
