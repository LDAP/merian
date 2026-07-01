#pragma once

#include "merian-shaders/utils/hash_grid.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <memory>

namespace merian {

class Properties;

class MCPG {
  public:
    MCPG(const ShaderCompileContextHandle& compile_context,
         const ResourceAllocatorHandle& allocator,
         uint32_t buffer_size);

    static SlangCompositionHandle query_device_support_composition();

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    void reset(const CommandBufferHandle& cmd) {
        grid.reset(cmd);
    }

    void write_to(ShaderCursor cursor) const {
        grid.write_to(cursor["grid"]);
    }

    void properties(Properties& props) {
        grid.properties(props);
    }

    uint32_t get_buffer_size() const {
        return grid.get_buffer_size();
    }

  private:
    SlangCompositionHandle composition;
    HashGrid grid;
};

using MCPGHandle = std::shared_ptr<MCPG>;

} // namespace merian
