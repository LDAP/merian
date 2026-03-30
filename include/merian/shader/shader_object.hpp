#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_object_layout.hpp"
#include "merian/shader/slang_utils.hpp"
#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
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

/**
 * @brief Represents a structured collection of shader parameters backed by Slang reflection.
 *
 * A ShaderObject maps to a Slang struct/type. It manages:
 * - Descriptor state (cached for incremental updates across frames)
 * - Ordinary data (uniform data uploaded via staging memory manager)
 * - Sub-objects (ConstantBuffer and ParameterBlock children, indexed by sub-object range)
 *
 * Sub-objects are stored in a flat array indexed by Slang's sub-object range index
 * (one slot per ConstantBuffer or ParameterBlock field in the type). This matches
 * the slang-rhi data model and avoids field_index ambiguity with value-embedded structs.
 *
 * Example:
 *   auto obj = std::make_shared<ShaderObject>(object_layout, allocator);
 *   obj->get_cursor()["my_texture"] = texture;
 *   obj->get_cursor()["my_value"] = 3.14f;
 *   obj->bind_as_parameter_block(cmd, pipeline, 0);
 */
class ShaderObject : public std::enable_shared_from_this<ShaderObject> {
  public:
    ShaderObject(const SlangObjectLayoutHandle& object_layout,
                 const ShaderObjectAllocatorHandle& allocator);

    ~ShaderObject();

    // ---------------------------------------------------------------
    // Binding

    /**
     * @brief Bind this object as a parameter block (its descriptor set) to the command buffer.
     *
     * - Gets a descriptor set from allocator
     * - If new: registers it and replays cached descriptor state
     * - Uploads ordinary data and ConstantBuffer sub-objects' data
     * - Writes nested ConstantBuffer descriptors to this set
     * - Flushes queued writes and binds
     *
     * This only binds this object, however ParameterBlock sub-objects are NOT bound here. Use
     * entry_point->bind_entry_point_parameter() to handle those (with reflection-derived set indices).
     */
    void bind_as_parameter_block(const CommandBufferHandle& cmd,
                                 const PipelineHandle& pipeline,
                                 uint32_t set_index);

    // ---------------------------------------------------------------
    // Cursor access

    /**
     * @brief Get a cursor pointing to the root of this object.
     *
     * Use this to write parameters: get_cursor()["field"] = value;
     */
    ShaderCursor get_cursor();

    // ---------------------------------------------------------------
    // Sub-object creation and access

    /**
     * @brief Create a new ShaderObject for a named ConstantBuffer or ParameterBlock field.
     *
     * Uses the pre-computed element layout from SlangObjectLayout.
     * The caller is responsible for assigning the result via set_sub_object(field_name, object).
     */
    ShaderObjectHandle create_sub_object(const std::string& field_name);

    /**
     * @brief Assign a ShaderObject as a sub-object for a named ConstantBuffer / ParameterBlock
     * field.
     *
     * Replaces any existing sub-object at that field. For ConstantBuffer fields,
     * also writes the object's buffer to this object's descriptor set.
     */
    void set_sub_object(const std::string& field_name, const ShaderObjectHandle& object);

    const ShaderObjectHandle& get_sub_object(uint32_t sub_object_range_index) const {
        assert(sub_object_range_index < sub_objects.size());
        return sub_objects[sub_object_range_index];
    }

    /**
     * @brief Set a sub-object by sub-object range index.
     *
     * For ConstantBuffer sub-objects, writes the CB's UBO descriptor to the owning PB's
     * descriptor storage (not the immediate parent's). This ensures descriptor writes happen
     * at update time, not bind time, enabling zero-write frames when nothing changes.
     */
    void set_sub_object(uint32_t sub_object_range_index, const ShaderObjectHandle& object);

    uint32_t get_sub_object_count() const {
        return static_cast<uint32_t>(sub_objects.size());
    }

    // ---------------------------------------------------------------
    // Write operations (called by ShaderCursor)

    void write(const ShaderOffset& offset, const ImageViewHandle& image);
    void write(const ShaderOffset& offset, const BufferHandle& buffer);
    void write(const ShaderOffset& offset, const TextureHandle& texture);
    void write(const ShaderOffset& offset, const SamplerHandle& sampler);
    void write(const ShaderOffset& offset, const void* data, std::size_t size);

    // ---------------------------------------------------------------
    // Accessors

    slang::TypeLayoutReflection* get_type_layout() const {
        return object_layout->get_type_layout();
    }

    const SlangObjectLayoutHandle& get_object_layout() const {
        return object_layout;
    }

    const ShaderObjectAllocatorHandle& get_allocator() const {
        return allocator;
    }

    // Print debug info
    std::string format_debug(const std::string& indent = "") const;

  private:
    // Recursively upload staging data for nested ConstantBuffer sub-objects.
    // Descriptor writes are handled at update time by set_sub_object.
    void upload_constant_buffer_tree(ShaderObject* cb_obj, const CommandBufferHandle& cmd);

    // Call fn on each live registered set, prune expired entries.
    void for_each_registered_set(const std::function<void(DescriptorContainer&)>& fn);

  private:
    SlangObjectLayoutHandle object_layout;
    ShaderObjectAllocatorHandle allocator;

    // Source of truth for all descriptor writes.
    // Replayed to the descriptor set at bind time.
    DescriptorStorageHandle descriptors;

    // Ordinary data (uniform buffer)
    BufferHandle ordinary_data_buffer;
    std::vector<uint8_t> ordinary_data_staging;
    bool ordinary_data_dirty = false;

    // Sub-objects indexed by sub-object range (one slot per ConstantBuffer/ParameterBlock field)
    std::vector<ShaderObjectHandle> sub_objects;

    // Registered descriptor sets for incremental write propagation.
    std::vector<std::weak_ptr<DescriptorContainer>> registered_sets;

    // For ConstantBuffer sub-objects: references to all owning ParameterBlocks and
    // the base binding offset for this CB's element content in each PB's descriptor set.
    // A CB can be shared across multiple PBs; each gets its own descriptor write.
    struct PBBinding {
        std::weak_ptr<ShaderObject> pb;
        uint32_t element_binding;
    };
    std::vector<PBBinding> pb_bindings_;

    friend class ShaderCursor;
    friend class SlangProgramEntryPoint;
};

} // namespace merian
