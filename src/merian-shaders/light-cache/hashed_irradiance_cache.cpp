#include "merian-shaders/light-cache/hashed_irradiance_cache.hpp"

namespace merian {

namespace {
constexpr const char* MODULE_PATH = "merian-shaders/light-cache/hashed-irradiance-cache.slang";

SlangCompositionHandle make_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(MODULE_PATH);
    return composition;
}
} // namespace

HashedIrradianceCache::HashedIrradianceCache(const ShaderCompileContextHandle& compile_context,
                                             const ResourceAllocatorHandle& allocator,
                                             const uint32_t buffer_size,
                                             const uint32_t probe_count,
                                             const bool stochastic_interpolation)
    : probe_count(probe_count), stochastic_interpolation(stochastic_interpolation),
      composition(make_composition()), grid(compile_context,
                                            allocator,
                                            composition,
                                            "merian::HashedIrradianceCacheData",
                                            buffer_size) {}

SlangCompositionHandle HashedIrradianceCache::query_device_support_composition() {
    return make_composition();
}

} // namespace merian
