#pragma once

#include "merian/shader/slang_utils.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

/**
 * @brief Cursor for navigating and writing shader parameter space.
 *
 * A cursor points at a ShaderOffset in a ShaderObject (TypeLayout).
 *
 * Write operations go through the ShaderObject, which handles broadcast
 * to all binding sites (other ShaderObjects where it's embedded).
 *
 * Example:
 *   cursor["material"]["roughness"] = 0.5f;
 */
class ShaderCursor {
  public:
    /**
     * @brief Create an invalid cursor
     */
    ShaderCursor() = default;

    /**
     * @brief Create a cursor pointing to the root of a ShaderObject.
     */
    ShaderCursor(const ShaderObjectHandle& base_object);

    // Navigation

    /**
     * @brief Navigate to a struct field by name.
     */
    ShaderCursor field(const std::string& name);

    /**
     * @brief Navigate to a struct field by index.
     */
    ShaderCursor field(uint32_t index);

    /**
     * @brief Navigate to an array element.
     */
    ShaderCursor element(uint32_t index);

    ShaderCursor operator[](const std::string& name) {
        return field(name);
    }

    /**
     * @brief Navigate to an array element or field index, depending on current location.
     */
    ShaderCursor operator[](uint32_t index);

    // Write operations

    ShaderCursor& write(const ImageViewHandle& image);
    ShaderCursor& write(const BufferHandle& buffer);
    ShaderCursor& write(const TextureHandle& texture);
    ShaderCursor& write(const SamplerHandle& sampler);
    ShaderCursor& write(const ShaderObjectHandle& object);
    ShaderCursor& write(const void* data, std::size_t size);

    template <class T> ShaderCursor& write(const T& data) {
        write(&data, sizeof(T));
        return *this;
    }

    // Assignment operators (explicit overloads must come before template)
    ShaderCursor& operator=(const ImageViewHandle& image) {
        return write(image);
    }
    ShaderCursor& operator=(const BufferHandle& buffer) {
        return write(buffer);
    }
    ShaderCursor& operator=(const TextureHandle& texture) {
        return write(texture);
    }
    ShaderCursor& operator=(const SamplerHandle& sampler) {
        return write(sampler);
    }
    ShaderCursor& operator=(const ShaderObjectHandle& object) {
        return write(object);
    }

    template <class T> ShaderCursor& operator=(const T& data) {
        return write(data);
    }

    // Query operations

    bool is_valid() const {
        return base_object != nullptr && type_layout != nullptr;
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

    bool is_struct() const {
        return get_kind() == slang::TypeReflection::Kind::Struct;
    }

    bool is_array() const {
        return get_kind() == slang::TypeReflection::Kind::Array;
    }

    bool is_resource() const {
        return get_kind() == slang::TypeReflection::Kind::Resource;
    }

    bool is_scalar() const {
        return get_kind() == slang::TypeReflection::Kind::Scalar;
    }

    bool is_vector() const {
        return get_kind() == slang::TypeReflection::Kind::Vector;
    }

    bool is_matrix() const {
        return get_kind() == slang::TypeReflection::Kind::Matrix;
    }

    // ---------------------------------------------------------------
    // Struct introspection

    /**
     * @brief Number of fields (for struct types).
     */
    uint32_t get_field_count() const {
        return type_layout->getFieldCount();
    }

    /**
     * @brief Get the name of a field by index.
     */
    const char* get_field_name(uint32_t index) const {
        assert(index < type_layout->getFieldCount());
        return type_layout->getFieldByIndex(index)->getVariable()->getName();
    }

    /**
     * @brief Get names of all fields.
     */
    std::vector<std::string> get_field_names() const;

    /**
     * @brief Check if a field with the given name exists.
     */
    bool has_field(const std::string& name) const {
        return type_layout->findFieldIndexByName(name.c_str()) >= 0;
    }

    /**
     * @brief Try to navigate to a field. Returns invalid cursor if not found.
     */
    std::optional<ShaderCursor> try_field(const std::string& name);

    // ---------------------------------------------------------------
    // Array introspection

    /**
     * @brief Number of elements (for array types).
     */
    uint32_t get_element_count() const {
        return static_cast<uint32_t>(type_layout->getElementCount());
    }

    /**
     * @brief Stride between array elements in bytes.
     */
    std::size_t get_element_stride() const {
        return type_layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    // ---------------------------------------------------------------
    // Size and type info

    /**
     * @brief Get the name of the type at this cursor.
     */
    const char* get_type_name() const {
        return type_layout->getName();
    }

    /**
     * @brief Get the uniform data size of the type at this cursor.
     */
    std::size_t get_uniform_size() const {
        return type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    /**
     * @brief Number of binding ranges (resource bindings) in this type.
     */
    uint32_t get_binding_range_count() const {
        return type_layout->getBindingRangeCount();
    }

    // ---------------------------------------------------------------
    // Debug

    // Print cursor position: type, kind, offset, field names
    std::string format_debug() const;

    // ---------------------------------------------------------------
    // Raw accessors

    slang::TypeLayoutReflection* get_type_layout() const {
        return type_layout;
    }

    const ShaderOffset& get_offset() const {
        return offset;
    }

    const ShaderObjectHandle& get_base_object() const {
        return base_object;
    }

  private:
    ShaderObjectHandle base_object;
    slang::TypeLayoutReflection* type_layout = nullptr;
    ShaderOffset offset;

    friend class ShaderObject;
};

} // namespace merian
