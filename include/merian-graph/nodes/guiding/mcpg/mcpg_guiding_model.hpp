#pragma once

#include "merian-graph/nodes/guiding/mcpg/mcpg.hpp"
#include "merian-shaders/light-cache/hashed_irradiance_cache.hpp"
#include "merian-shaders/sampling/guiding.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

// Markov-chain path guiding over an adaptive hash grid, learning from an irradiance cache.
class MCPGGuidingModel : public GuidingModel {
  public:
    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    static SlangCompositionHandle query_device_support_composition();

    SlangCompositionHandle get_composition() const override;

    std::vector<std::string> get_slang_imports() const override;

    std::string get_type_name() const override;

    void write_to(ShaderCursor cursor) override;

    void reset(const CommandBufferHandle& cmd) override;

    bool properties(Properties& props) override;

  private:
    void recreate_grids();

    ShaderCompileContextHandle compile_context;
    ResourceAllocatorHandle allocator;

    MCPGHandle mcpg;
    HashedIrradianceCacheHandle irr_cache;

    uint32_t mc_buffer_size = 32777259;
    uint32_t mc_probe_count = 2;
    uint32_t mc_locality_bits = 3;
    bool mc_split_storage = true;

    uint32_t lc_buffer_size = 4000000;
    uint32_t lc_probe_count = 4;
    uint32_t lc_locality_bits = 3;
    bool lc_stochastic_interpolation = false;
    bool lc_split_storage = true;

    int32_t mc_samples = 5;
    float dir_guide_prior = 0.2f;
    float probability = 0.5f;
    bool scale_with_alpha = true;
    float alpha_threshold = 0.05f;
    bool missing_light_heuristic = true;
    bool nee_reweight = false;
    bool light_cache_tail = false;
    float lc_min_pdf = 1.0f;
};

} // namespace merian
