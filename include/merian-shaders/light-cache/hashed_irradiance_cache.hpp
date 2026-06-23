#pragma once

#include "merian-shaders/light-cache/light-cache.slangh"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <memory>

namespace merian {

class Properties;

class HashedIrradianceCache : public std::enable_shared_from_this<HashedIrradianceCache> {
  public:
    HashedIrradianceCache(const ResourceAllocatorHandle& allocator,
                          uint32_t buffer_size,
                          uint32_t probe_count = 4,
                          bool stochastic_interpolation = false);

    static SlangCompositionHandle query_device_support_composition();

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    void reset(const CommandBufferHandle& cmd);

    void write_to(ShaderCursor cursor) const;

    void properties(Properties& props);

    uint32_t get_buffer_size() const {
        return buffer_size;
    }
    uint32_t get_probe_count() const {
        return probe_count;
    }
    bool get_stochastic_interpolation() const {
        return stochastic_interpolation;
    }

  private:
    ResourceAllocatorHandle allocator;

    const uint32_t buffer_size;
    const uint32_t probe_count;
    const bool stochastic_interpolation;

    float grid_tan_alpha_half = 0.006F;
    float grid_steps_per_unit_size = 2.0F;
    float grid_min_width = 0.001F;
    float grid_power = 2.0F;

    BufferHandle buffer;
    SlangCompositionHandle composition;
};

using HashedIrradianceCacheHandle = std::shared_ptr<HashedIrradianceCache>;

} // namespace merian
