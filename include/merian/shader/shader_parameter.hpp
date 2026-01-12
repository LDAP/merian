#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "slang.h"

#include <cstddef>
#include <cstdint>

namespace merian {

/*
 * - Nest as value: Cannot be reused -> object is only used as part of the outer struct (e.g.
 * just for logical aggregation but not sharing) and cannot be shared between structs or shaders.
 * Member variable on Host. The same object will never be also bound as constant buffer or parameter
 * block.
 * => The nested object does not need to be smart about making sure descriptor and buffer updates
 * are executed, its the outer objects reponsibility. Also at max just one cursor (the outer one) is
 * needed since its position is fixed.
 *
 * - Nest as constant buffer: The constant buffer can be reused (and thus has to be updated only
 * once). For a small reused object that does not need its own descriptor set and is used only as
 * part of the outer struct in the shader (struct of constants about something).
 *
 * - Nest as parameter block: The whole object might be reused at other places and always looks the
 * same in the shader (same struct definition, same descriptor set and layout can be used). It is
 * independent of the outer object. Has its own constant buffer and descriptor set (index). For
 * example TextureManager, Scene.
 * => The nested object does not need any reference to the outer object since it records its own
 * updates and only when binding (the set) to the actual shader the outer object calls into the
 * nested one to do so.
 *
 */

struct BindingOffset {
    uint32_t binding_set;
    uint32_t binding;
    // uint32_t array_element;

    uint32_t push_constant_range = 0;
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

class IParameterBlock {

  private:
    DescriptorContainerHandle descriptors;
};

struct ShaderObjectLayout {
    DescriptorSetLayoutHandle descriptor_set_layout;
    std::size_t ordinary_buffer_size;
};

// This represents a Slang-like struct that is bound to a shader either directly to a ParameterBlock
// or as member of another ShaderObject as value or ConstantBuffer.
class ShaderObject {
  public:
    void bind_as_value(ShaderCursor& cursor);

    void bind_as_constant_buffer(ShaderCursor& cursor);

    void bind_as_parameter_pack(ShaderCursor& cursor);

  private:
    // Contains the ordinary data of this object and all objects that are value members of this
    // object.
    //
    // Can be nullptr if this object was only bound as value to parents (then their ordinary data
    // buffer is used) Do not write to this buffer directly but use the cursor in the binding
    // instead.
    BufferHandle ordinary_data = nullptr;

    // the cursors pointing to the offsets of this object in parameter blocks this object is bound
    // to.
    std::vector<ShaderCursor> bindings;
};

} // namespace merian
