#pragma once

// see
// https://docs.shader-slang.org/en/latest/shader-cursors.html#making-a-multi-platform-shader-cursor
//
// this is the ShaderObject in slang documentation.
//
// It holds the buffer and descriptor set for one feature, i.e a ParameterBlock<...> and the
// target specific functions to write into it.

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "slang.h"

#include <cstddef>
#include <cstdint>

namespace merian {

struct ShaderOffset {
    std::size_t byte_offset = 0;
    uint32_t binding_range_offset = 0;
    uint32_t binding_array_index = 0;
};

class ShaderParameterBlock {
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

class DescriptorContainerShaderParameterBlock : public ShaderParameterBlock {
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

} // namespace merian
