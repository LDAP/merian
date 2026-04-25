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
 * @brief Lightweight cursor for navigating and writing shader parameter space.
 *
 * The ShaderObject must outlive the cursor.
 */
class ShaderCursor {
  public:
    /**
     * @brief Create an invalid cursor
     */
    ShaderCursor() = default;

    /**
     * @brief Create a cursor pointing to the root of a ShaderObject.
     *
     * Uses a raw pointer — the caller must ensure the ShaderObject outlives the cursor.
     */
    ShaderCursor(ShaderObject* base_object);

    // Navigation

    /**
     * @brief Navigate to a struct field by name.
     */
    ShaderCursor field(const std::string& name);

    /**
     * @brief Navigate to a struct field by name.
     */
    ShaderCursor operator[](const std::string& name) {
        return field(name);
    }

    /**
     * @brief Navigate to an array element, matrix, vector or field index, depending on current
     * location.
     */
    ShaderCursor element(uint32_t index);

    /**
     * @brief Navigate to an array element, matrix, vector or field index, depending on current
     * location.
     */
    ShaderCursor operator[](uint32_t index);

    // Write operations

    ShaderCursor& write(const ImageViewHandle& image,
                        const std::optional<vk::ImageLayout> access_layout = std::nullopt);
    ShaderCursor& write(const BufferHandle& buffer);
    ShaderCursor& write(const TextureHandle& texture,
                        const std::optional<vk::ImageLayout> access_layout = std::nullopt);
    ShaderCursor& write(const SamplerHandle& sampler);
    ShaderCursor& write(const AccelerationStructureHandle& as);
    ShaderCursor& write(const void* data, std::size_t size);
    template <class T> ShaderCursor& write(const T& data) {
        write(&data, sizeof(T));
        return *this;
    }
    ShaderCursor& write(const ShaderObjectHandle& object);

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
    ShaderCursor& operator=(const AccelerationStructureHandle& as) {
        return write(as);
    }
    template <class T> ShaderCursor& operator=(const T& data) {
        return write(data);
    }
    ShaderCursor& operator=(const ShaderObjectHandle& object) {
        return write(object);
    }

    // Accept shared_ptr<T> where T provides operator const ShaderObjectHandle&()
    template <class T>
        requires(!std::is_same_v<T, ShaderObject> &&
                 std::is_convertible_v<const T&, const ShaderObjectHandle&>)
    ShaderCursor& operator=(const std::shared_ptr<T>& object) {
        return write(static_cast<const ShaderObjectHandle&>(*object));
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

    // ---------------------------------------------------------------
    // Array / Vector / Matrix introspection

    /**
     * @brief Number of elements (for array, vector, matrix vector types).
     */
    uint32_t get_element_count() const {
        return static_cast<uint32_t>(type_layout->getElementCount());
    }

    /**
     * @brief Stride between array, vector, matrix elements in bytes.
     */
    std::size_t get_element_stride() const {
        return type_layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    /**
     * @brief TypeLayout of array, vector, matrix elements.
     */
    slang::TypeLayoutReflection* get_element_type_layout() const {
        return type_layout->getElementTypeLayout();
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
    // Raw accessors

    slang::TypeLayoutReflection* get_type_layout() const {
        return type_layout;
    }

    const ShaderOffset& get_offset() const {
        return offset;
    }

    ShaderObject* get_base_object() const {
        return base_object;
    }

  private:
    /**
     * @brief Navigate to a struct field by index.
     */
    ShaderCursor field(uint32_t index);

    /**
     * @brief Dereference a ParameterBlock/ConstantBuffer cursor into its element type,
     * auto-creating the subobject if needed.
     */
    ShaderCursor dereference();

  private:
    ShaderObject* base_object = nullptr;
    slang::TypeLayoutReflection* type_layout = nullptr;
    ShaderOffset offset;

    friend class ShaderObject;
};

std::string format_as(const ShaderCursor& cursor);

} // namespace merian
