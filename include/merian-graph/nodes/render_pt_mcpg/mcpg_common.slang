__exported import merian_shaders.light_cache.hashed_irradiance_cache;
__exported import mc;

using merian;

namespace merian {
extern static const uint merian_render_instance_mask = 0xFFu;
}

extern static const uint lc_buffer_size = 4000000;
extern static const uint lc_probe_count = 4;
extern static const bool lc_stochastic_interpolation = false;
extern static const bool lc_split_storage = true;
extern static const uint lc_locality_bits = 3;

extern static const uint mc_adaptive_buffer_size = 32777259;
extern static const uint mc_probe_count = 2;
extern static const bool mc_split_storage = true;
extern static const uint mc_locality_bits = 3;

extern static const int debug_output_selector = 0;
extern static const int mc_samples = 5;
extern static const bool missing_light_heuristic = true;

// The surface and the volume pass share both grids, so they must agree on the instantiation.
typealias RenderIrradianceCache =
    HashedIrradianceCache<lc_buffer_size, lc_probe_count, lc_stochastic_interpolation,
                          lc_split_storage, lc_locality_bits>;
typealias RenderMCPG =
    MCPG<mc_adaptive_buffer_size, mc_probe_count, mc_split_storage, mc_locality_bits>;
