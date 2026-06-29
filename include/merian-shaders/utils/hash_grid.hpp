#pragma once

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <string>

namespace merian {

class Properties;

class HashGrid {
  public:
    HashGrid(const ShaderCompileContextHandle& compile_context,
             const ResourceAllocatorHandle& allocator,
             const SlangCompositionHandle& composition,
             const std::string& data_type_name,
             uint32_t buffer_size);

    void reset(const CommandBufferHandle& cmd);

    void write_to(ShaderCursor cursor) const;

    void properties(Properties& props);

    uint32_t get_buffer_size() const {
        return buffer_size;
    }

    const BufferHandle& get_buffer() const {
        return buffer;
    }

  private:
    const uint32_t buffer_size;

    float grid_tan_alpha_half = 0.006F;
    float grid_min_width = 0.001F;

    BufferHandle buffer;
};

} // namespace merian
