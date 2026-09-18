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
    // The grid's resolution, kept apart from the buffers so that it survives a rebuild.
    struct Params {
        float tan_alpha_half = 0.006F;
        float level_bias = 0.0F;
        float distribution_dimension = 2.0F;

        void properties(Properties& props);
    };

    HashGrid(const ShaderCompileContextHandle& compile_context,
             const ResourceAllocatorHandle& allocator,
             const SlangCompositionHandle& composition,
             const std::string& data_type_name,
             uint32_t buffer_size,
             bool split_storage = true);

    void reset(const CommandBufferHandle& cmd);

    void write_to(ShaderCursor cursor) const;

    void set_params(const Params& p) {
        params = p;
    }

    uint32_t get_buffer_size() const {
        return buffer_size;
    }

    bool get_split_storage() const {
        return split_storage;
    }

  private:
    const uint32_t buffer_size;
    const bool split_storage;

    Params params;

    BufferHandle keys;
    BufferHandle data;
};

} // namespace merian
