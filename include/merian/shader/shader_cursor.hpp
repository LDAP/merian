#pragma once

#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_utils.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

/**
 * @brief Cursor for navigating and writing shader parameter space.
 *
 * All cursors are implicitly multi-cursors, containing a list of locations.
 * Navigation and write operations traverse all locations automatically.
 *
 * Example:
 *   cursor["material"]["roughness"] = 0.5f;
 * // If cursor tracks 3 locations, all 3 get updated
 */
class ShaderCursor {
  public:
    ShaderCursor() = default;

    /**
     * @brief Create a cursor with a single location.
     *
     * @param base_object The object this cursor points into
     * @param type_layout Slang type layout at this position
     * @param offset Offset within the object
     */
    ShaderCursor(const ShaderObjectHandle& base_object);

    // Navigation - all operations traverse all locations

    /**
     * @brief Navigate to a struct field by name.
     *
     * @param name Field name
     * @return New cursor pointing to the field in all locations
     */
    ShaderCursor field(const std::string& name);

    /**
     * @brief Navigate to a struct field by index.
     *
     * @param index Field index
     * @return New cursor pointing to the field in all locations
     */
    ShaderCursor field(uint32_t index);

    /**
     * @brief Navigate to an array element.
     *
     * @param index Element index
     * @return New cursor pointing to the element in all locations
     */
    ShaderCursor element(uint32_t index);

    ShaderCursor operator[](const std::string& name) {
        return field(name);
    }
    ShaderCursor operator[](uint32_t index);

    // Write operations - traverse all locations

    ShaderCursor& write(const ImageViewHandle& image);
    ShaderCursor& write(const BufferHandle& buffer);
    ShaderCursor& write(const TextureHandle& texture);
    ShaderCursor& write(const SamplerHandle& sampler);
    ShaderCursor& write(const void* data, std::size_t size);

    template <class T> ShaderCursor& write(const T& data) {
        write(&data, sizeof(T));
        return *this;
    }

    // Assignment operators
    ShaderCursor& operator=(const ImageViewHandle& image) {
        write(image);
        return *this;
    }
    ShaderCursor& operator=(const BufferHandle& buffer) {
        write(buffer);
        return *this;
    }
    ShaderCursor& operator=(const TextureHandle& texture) {
        write(texture);
        return *this;
    }
    ShaderCursor& operator=(const SamplerHandle& sampler) {
        write(sampler);
        return *this;
    }

    template <class T> ShaderCursor& operator=(const T& data) {
        write(data);
        return *this;
    }

    /**
     * @brief Bind a nested shader object at this cursor position.
     *
     * Depending on the type (parameter block, constant buffer, value),
     * the object will be bound appropriately.
     *
     * @param object The object to bind
     * @param allocator Allocator for descriptor sets (if needed)
     */
    void bind_object(const ShaderObjectHandle& object, ShaderObjectAllocator& allocator);

    // Query operations

    bool is_valid() const {
        return !locations.empty() && type_layout != nullptr;
    }
    bool is_empty() const {
        return locations.empty();
    }
    size_t location_count() const {
        return locations.size();
    }

    slang::TypeReflection::Kind get_kind() const {
        return type_layout->getType()->getKind();
    }

    bool is_parameter_block() const {
        return get_kind() == slang::TypeReflection::Kind::ParameterBlock;
    }

    bool is_constant_buffer() const {
        return get_kind() == slang::TypeReflection::Kind::ConstantBuffer;
    }

    /**
     * @brief Add locations from another cursor to this one.
     *
     * Used when an object is bound in multiple places.
     *
     * @param other The cursor whose locations to add
     */
    void add_locations(const ShaderCursor& other);

  private:
    struct Location {
        ShaderObjectHandle base_object;
        ShaderOffset offset;
    };

    std::vector<Location> locations;
    slang::TypeLayoutReflection* type_layout = nullptr;

    friend class ShaderObject;
};

} // namespace merian
