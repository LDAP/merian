#pragma once

#include "merian-shaders/light-cache/light-cache.slangh"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <memory>

namespace merian {

class Properties;

// Hashed, multi-level irradiance cache backed by a single persistent storage buffer. The buffer is
// updated in place across frames; call reset() to zero it at the start of an accumulation run.
class HashedIrradianceCache : public std::enable_shared_from_this<HashedIrradianceCache> {
  public:
    HashedIrradianceCache(const ShaderCompileContextHandle& compile_context,
                          const ContextHandle& context,
                          const ResourceAllocatorHandle& allocator,
                          uint32_t capacity = 4000000);

    static SlangCompositionHandle query_device_support_composition();

    // Allocates the buffer on the first call (and after a capacity change) and keeps the shader
    // object current. Call once per frame before binding.
    void update(const CommandBufferHandle& cmd);

    // Zero the cache. Call on the first frame of an accumulation run.
    void reset(const CommandBufferHandle& cmd);

    void properties(Properties& props);

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    // Bumps whenever the composition changes.
    uint64_t version() const {
        return composition->version();
    }

    const ShaderObjectHandle& get_shader_object() const {
        return shader_object.get();
    }

    operator const ShaderObjectHandle&() const {
        return shader_object.get();
    }

  private:
    void update_composition_constants();
    ShaderObjectHandle build_shader_object() const;

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;

    uint32_t capacity;
    float lc_grid_tan_alpha_half = 0.006F;
    float lc_grid_steps_per_unit_size = 2.0F;
    float lc_grid_min_width = 0.001F;
    float lc_grid_power = 2.0F;

    BufferHandle buffer;
    bool needs_realloc = true;

    SlangCompositionHandle composition;
    Versioned<SlangProgram> layout_program;
    Versioned<ShaderObject> shader_object;
};

using HashedIrradianceCacheHandle = std::shared_ptr<HashedIrradianceCache>;

} // namespace merian
