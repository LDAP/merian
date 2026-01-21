#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_utils.hpp"
#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "slang.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace merian {

/*
 * Shader parameter system based on Slang reflection API and shader cursors.
 *
 * Key concepts:
 * - ShaderObject: Represents shader parameters (uniform data + resources)
 * - ShaderCursor: Points to position(s) in shader parameter space
 * - All cursors are implicitly multi-cursors (list of locations)
 * - When an object is bound in multiple places, its cursor tracks all locations
 * - Updates automatically propagate to all binding locations
 *
 * Binding modes:
 * - Parameter Block: Object gets its own descriptor set (e.g., ParameterBlock<T>)
 * - Constant Buffer: Object gets its own buffer, bound to parent's descriptor set
 * - Value: Object's data is embedded in parent's buffer/descriptor set
 */

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

/**
 * @brief Base class for shader parameter objects.
 *
 * Represents a structured collection of shader parameters that can be bound
 * as parameter blocks, constant buffers, or embedded values.
 *
 * Each object maintains a root cursor that tracks all binding locations.
 * Updates through the cursor automatically propagate to all locations.
 */
class ShaderObject : public std::enable_shared_from_this<ShaderObject> {

  public:
    struct ParameterBlock {
        // Ordinary data buffer (for uniform data) of this object and all objects that are value
        // members of this object.
        //
        // Can be nullptr if this object was only bound as value to parents (then their ordinary
        // data buffer is used). Do not write to this buffer directly but use the cursor in the
        // binding instead.
        BufferHandle ordinary_data = nullptr;
        std::vector<uint8_t> ordinary_data_staging;

        // All descriptor sets that should be updated whenever this object changes
        // Only non-empty if were used as parameter block somewhere. Do not write to these sets
        // directly but use the cursor in the binding instead.
        std::set<std::weak_ptr<DescriptorContainer>, std::owner_less<DescriptorContainerHandle>>
            descriptor_sets;
    };

  public:
    ShaderObject(const ContextHandle& context, slang::TypeLayoutReflection* type_layout);

    virtual ~ShaderObject() = default;

    /**
     * @brief Initialize this object as a parameter block.
     *
     * Creates a descriptor set and ordinary data buffer (if needed).
     * The object can then be bound to different pipelines at different set indices.
     *
     * @param allocator Allocator for descriptor sets
     * @return The descriptor set handle (for binding to command buffer)
     */
    DescriptorContainerHandle initialize_as_parameter_block(ShaderObjectAllocator& allocator);

    /**
     * @brief Bind this object to a cursor position.
     *
     * Depending on the cursor's type (parameter block, constant buffer, or value),
     * this will either create a new descriptor set or merge into the parent's resources.
     *
     * @param cursor The cursor indicating where to bind
     * @param allocator Allocator for descriptor sets (if needed)
     */
    void bind_to(ShaderCursor& cursor, ShaderObjectAllocator& allocator);

    /**
     * @brief Populate this object's parameters through a cursor.
     *
     * Subclasses override this to write their data to the shader.
     * This is called during initialization and when binding.
     *
     * @param cursor The cursor to write data through
     */
    virtual void populate(ShaderCursor& cursor) = 0;

    /**
     * @brief Get the root cursor for this object.
     *
     * This cursor tracks all locations where this object is bound.
     * Writing through this cursor updates all binding locations.
     *
     * @return Reference to the root cursor
     */
    ShaderCursor& get_cursor() {
        return *root_cursor;
    }
    const ShaderCursor& get_cursor() const {
        return *root_cursor;
    }

    /**
     * @brief Get the descriptor sets and ordinary data buffer as parameter block.
     *
     * @return the parameter block. The members can be empty or nullptr respectivly if there is no
     * data buffer or this object was never used as parameter block.
     */
    const ParameterBlock& get_parameter_block() const {
        return parameter_block;
    }

    // Write operations - called by cursors
    void write(const ShaderOffset& offset, const ImageViewHandle& image);
    void write(const ShaderOffset& offset, const BufferHandle& buffer);
    void write(const ShaderOffset& offset, const TextureHandle& texture);
    void write(const ShaderOffset& offset, const SamplerHandle& sampler);
    void write(const ShaderOffset& offset, const void* data, std::size_t size);

    slang::TypeLayoutReflection* get_type_layout() const {
        return type_layout;
    }
    const ContextHandle& get_context() const {
        return context;
    }

  private:
    void for_each_descriptor_set(const std::function<void(const DescriptorContainerHandle&)>& f);

  private:
    // The root cursor - tracks all binding locations for this object
    std::optional<ShaderCursor> root_cursor;

    ParameterBlock parameter_block;

    slang::TypeLayoutReflection* type_layout;
    ContextHandle context;

    friend class ShaderCursor;
};

} // namespace merian
