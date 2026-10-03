#include "merian-shaders/light-cache/hashed_irradiance_cache.hpp"

#include <fmt/format.h>

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
                                             const bool stochastic_interpolation,
                                             const bool split_storage,
                                             const uint32_t locality_bits)
    : probe_count(probe_count), stochastic_interpolation(stochastic_interpolation),
      locality_bits(locality_bits), composition(make_composition()),
      grid(compile_context,
           allocator,
           composition,
           "merian::HashedIrradianceCacheData",
           buffer_size,
           split_storage) {}

SlangCompositionHandle HashedIrradianceCache::query_device_support_composition() {
    return make_composition();
}

std::vector<std::string> HashedIrradianceCache::get_slang_imports() const {
    return {slang_import_spelling(MODULE_PATH)};
}

std::string HashedIrradianceCache::get_type_name() const {
    return fmt::format("merian::HashedIrradianceCache<{}u, {}u, {}, {}, {}u>", get_buffer_size(),
                       probe_count, stochastic_interpolation ? "true" : "false",
                       grid.get_split_storage() ? "true" : "false", locality_bits);
}

} // namespace merian
