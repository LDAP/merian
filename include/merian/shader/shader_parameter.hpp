#pragma once

// see
// https://docs.shader-slang.org/en/latest/shader-cursors.html#making-a-multi-platform-shader-cursor
// this is the ShaderCursor from Slangs docs.

#include "merian/shader/shader_parameter_block.hpp"

#include "slang.h"

namespace merian {

// Points to a position in a shader. Positions can be structs, fields (ordinary data, opaque data),
// array (elemenets).
class ShaderCursor {
  public:
    // --------------------------------------------------------------------

    ShaderCursor field(const std::string& name) {
        return field(type_layout->findFieldIndexByName(name.c_str()));
    }

    ShaderCursor field(const uint32_t index) {
        assert(index < type_layout->getFieldCount());

        slang::VariableLayoutReflection* field = type_layout->getFieldByIndex(index);

        ShaderCursor result = *this;
        result.type_layout = field->getTypeLayout();
        result.offset.uniform_byte_offset += field->getOffset();
        result.offset.binding_range_offset += type_layout->getFieldBindingRangeOffset(index);

        return result;
    }

    ShaderCursor element(const uint32_t index) {
        slang::TypeLayoutReflection* element_type_layout = type_layout->getElementTypeLayout();

        ShaderCursor result = *this;
        result.type_layout = element_type_layout;
        result.offset.uniform_byte_offset += index * element_type_layout->getStride();

        result.offset.binding_array_index *= type_layout->getElementCount();
        result.offset.binding_array_index += index;

        return result;
    }

    ShaderCursor operator[](const std::string& name) {
        return field(name);
    }

    ShaderCursor operator[](const uint32_t index) {
        return element(index);
    }

    // ShaderCursor dereference() {

    // }

    // --------------------------------------------------------------------

    ShaderCursor& write(const ImageHandle& image) {
        parameter_block->write(offset, image);
        return *this;
    }

    ShaderCursor& write(const BufferHandle& buffer) {
        parameter_block->write(offset, buffer);
        return *this;
    }

    ShaderCursor& write(const TextureHandle& texture) {
        parameter_block->write(offset, texture);
        return *this;
    }

    ShaderCursor& write(const SamplerHandle& sampler) {
        parameter_block->write(offset, sampler);
        return *this;
    }

    ShaderCursor& write(const void* data, std::size_t size) {
        parameter_block->write(offset, data, size);
        return *this;
    }

    template <class T> ShaderCursor& write(const T& data) {
        write(&data, sizeof(T));
        return *this;
    }

    ShaderCursor& operator=(const ImageHandle& image) {
        parameter_block->write(offset, image);
        return *this;
    }

    ShaderCursor& operator=(const BufferHandle& buffer) {
        parameter_block->write(offset, buffer);
        return *this;
    }

    ShaderCursor& operator=(const TextureHandle& texture) {
        parameter_block->write(offset, texture);
        return *this;
    }

    ShaderCursor& operator=(const SamplerHandle& sampler) {
        parameter_block->write(offset, sampler);
        return *this;
    }

    template <class T> ShaderCursor& operator=(const T& data) {
        write(&data, sizeof(T));
        return *this;
    }

  private:
    ShaderParameterBlock* parameter_block = nullptr;

    slang::TypeLayoutReflection* type_layout;
    ShaderOffset offset;
};

} // namespace merian
