#pragma once

#include "merian-shaders/utils/hash_grid.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <memory>

namespace merian {

class Properties;

class HashedIrradianceCache {
  public:
    HashedIrradianceCache(const ShaderCompileContextHandle& compile_context,
                          const ResourceAllocatorHandle& allocator,
                          uint32_t buffer_size,
                          uint32_t probe_count = 4,
                          bool stochastic_interpolation = false);

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
    uint32_t get_probe_count() const {
        return probe_count;
    }
    bool get_stochastic_interpolation() const {
        return stochastic_interpolation;
    }

  private:
    const uint32_t probe_count;
    const bool stochastic_interpolation;

    SlangCompositionHandle composition;
    HashGrid grid;
};

using HashedIrradianceCacheHandle = std::shared_ptr<HashedIrradianceCache>;

} // namespace merian
