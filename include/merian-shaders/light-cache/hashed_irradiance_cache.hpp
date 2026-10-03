#pragma once

#include "merian-shaders/light-cache/irradiance_cache.hpp"
#include "merian-shaders/utils/hash_grid.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <memory>

namespace merian {

class Properties;

class HashedIrradianceCache : public IrradianceCache {
  public:
    HashedIrradianceCache(const ShaderCompileContextHandle& compile_context,
                          const ResourceAllocatorHandle& allocator,
                          uint32_t buffer_size,
                          uint32_t probe_count = 4,
                          bool stochastic_interpolation = false,
                          bool split_storage = true,
                          uint32_t locality_bits = 0);

    static SlangCompositionHandle query_device_support_composition();

    SlangCompositionHandle get_composition() const override {
        return composition;
    }

    std::vector<std::string> get_slang_imports() const override;

    std::string get_type_name() const override;

    void write_to(ShaderCursor cursor) override {
        grid.write_to(cursor["grid"]);
    }

    void reset(const CommandBufferHandle& cmd) {
        grid.reset(cmd);
    }

    void set_grid_params(const HashGrid::Params& p) {
        grid.set_params(p);
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

    void set_locality_bits(const uint32_t bits) {
        locality_bits = bits;
    }

  private:
    const uint32_t probe_count;
    const bool stochastic_interpolation;
    uint32_t locality_bits;

    SlangCompositionHandle composition;
    HashGrid grid;
};

using HashedIrradianceCacheHandle = std::shared_ptr<HashedIrradianceCache>;

} // namespace merian
