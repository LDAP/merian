#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "slang.h"

#include <cstddef>
#include <cstdint>

namespace merian {

struct BindingOffset {
    uint32_t binding_set;
    uint32_t binding;
    // uint32_t array_element;
};

struct ShaderOffset {
    std::size_t uniform_byte_offset = 0;
    uint32_t binding_range_offset = 0;
    uint32_t binding_array_index = 0;
};

// see
// https://docs.shader-slang.org/en/latest/shader-cursors.html#making-a-multi-platform-shader-cursor
//
// this is the ShaderObject in slang documentation.
//
// It holds the buffer and descriptor set for one feature, i.e a ParameterBlock<...> or the default
// toplevel "block" and the target specific functions to write into it.
class ParameterBlock {
  public:
    virtual void write(const ShaderOffset& offset, const ImageHandle& image) = 0;

    virtual void write(const ShaderOffset& offset, const BufferHandle& buffer) = 0;

    virtual void write(const ShaderOffset& offset, const TextureHandle& texture) = 0;

    virtual void write(const ShaderOffset& offset, const SamplerHandle& sampler) = 0;

    virtual void write(const ShaderOffset& offset, const void* data, std::size_t size) = 0;

    template <class T> void write(const ShaderOffset& offset, const T& data) {
        write(offset, &data, sizeof(T));
    }
};

using ParameterBlockHandle = std::shared_ptr<ParameterBlock>;

class DescriptorContainerParameterBlock : public ParameterBlock {
  public:
    virtual void write(const ShaderOffset& offset, const ImageHandle& image) override {}

    virtual void write(const ShaderOffset& offset, const BufferHandle& buffer) override {}

    virtual void write(const ShaderOffset& offset, const TextureHandle& texture) override {}

    virtual void write(const ShaderOffset& offset, const SamplerHandle& sampler) override {}

    virtual void write(const ShaderOffset& offset, const void* data, std::size_t size) override {}

  private:
    slang::TypeLayoutReflection* type_layout;

    

    BufferHandle constant_buffer;
    DescriptorContainerHandle descriptor_container;
};

// Points to a position in a shader. Positions can be structs, fields (ordinary data, opaque data),
// array (elemenets). Always relative to the corresponding parameter block this cursor refers to.
//
// see
// https://docs.shader-slang.org/en/latest/shader-cursors.html#making-a-multi-platform-shader-cursor
// this is the ShaderCursor from Slangs docs.
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
    ParameterBlockHandle parameter_block = nullptr;

    slang::TypeLayoutReflection* type_layout;
    ShaderOffset offset;
};

// This represents a Slang-like struct that is bound to a shader either directly to a ParameterBlock
// or as member of another ShaderObject.
class ShaderObject {
  public:
    // Binds this object to the parameter block and starts updating the parameter block whenever
    // this object changes.
    void bind(ParameterBlock& block, ShaderCursor& cursor);

  private:
    // all the parameter blocks this object is currently bound to as well as a cursor that points to
    // the object.
    std::vector<std::pair<ParameterBlockHandle, ShaderCursor>> bindings;
};

} // namespace merian
