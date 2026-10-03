#pragma once

#include "merian-shaders/scene/light-data.slangh"
#include "merian-shaders/scene/light-select.slangh"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/versioned.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace merian {

class Properties;

// The emissive triangles of a Scene and the flux-proportional sampling data behind merian::NEE
// (nee.slang). Records are rebuilt on the GPU every update from the scene's geometry and
// materials, so animation, morphing and material edits need no bookkeeping.
class LightCollection {
  public:
    struct EmissiveGeometry {
        GeometryID geometry_id;
        uint32_t instance_index;
        uint32_t primitive_count;
        vk::DeviceAddress emission_offsets = 0;
        vk::DeviceAddress emission_records = 0;
        uint32_t emission_level = 0;
    };

    LightCollection(const ShaderCompileContextHandle& compile_context,
                    const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator);

    // Emissive geometries of the frame in GeometryID order; geometry_count sizes the lookup table.
    void set_geometries(const std::vector<EmissiveGeometry>& geometries, uint32_t geometry_count);

    // Allocates and uploads the tables; call before write_to so the cursor sees this frame's
    // buffers.
    void prepare(const CommandBufferHandle& cmd);

    // Records the update passes. The scene object must already hold this frame's geometry,
    // transforms and materials and must not be written afterwards (it is bound here).
    void update_constants(const SlangCompositionHandle& scene_composition);

    void update(const CommandBufferHandle& cmd,
                const SlangCompositionHandle& scene_composition,
                const ShaderObjectHandle& scene_object,
                const ShaderObjectAllocatorHandle& obj_allocator);

    // Binds the buffers to a merian::NEE cursor.
    void write_to(ShaderCursor cursor) const;

    void properties(Properties& props);

    uint32_t get_triangle_count() const {
        return triangle_count;
    }

    uint32_t get_geometry_count() const {
        return static_cast<uint32_t>(light_geometries.size());
    }

    bool get_enabled() const {
        return enabled;
    }

    void set_enabled(const bool value) {
        enabled = value;
    }

    // The manual environment share, used where it is not derived from the emitted power.
    float get_env_share() const {
        return env_share;
    }

    void set_env_share(const float p) {
        env_share = p;
    }

    // Bounding sphere the environment's power is measured through; 0 keeps the manual share.
    void set_scene_radius(const float radius) {
        scene_radius = radius;
    }

    // Whether the scene's environment map emits; false routes all samples to the triangles.
    // The importance map is a function of the environment alone, so a static one is built once
    // and kept until it changes. The per-frame texel pool is drawn from it either way.
    void set_env_state(const uint64_t version, const bool is_static) {
        if (version != env_version || !is_static) {
            env_importance_built = false;
        }
        env_version = version;
    }

    void set_env_emissive(const bool value) {
        env_emissive = value;
    }

    // The grid follows the camera.
    void set_camera(const float3& position) {
        camera_position = position;
    }

    void set_frame(const uint32_t frame_index) {
        frame = frame_index;
    }

    // Whether UseEnvMap geometry exists (rays to the environment must not stop at it).
    void set_has_sky_portals(const bool value) {
        has_sky_portals = value;
    }

  private:
    void ensure_pipelines(const SlangCompositionHandle& scene_composition);
    uint32_t env_importance_size() const {
        return 1u << static_cast<uint32_t>(env_importance_log2);
    }
    bool lists_env() const {
        return grid_lists_env && env_emissive && env_split_buffer && env_cdf_buffer;
    }
    // threads per block of the scans; matches CDF_GROUP_SIZE
    static constexpr uint32_t CDF_GROUP_SIZE = 1024;
    static uint32_t cdf_block_count(const uint32_t count) {
        return (count + CDF_GROUP_SIZE - 1) / CDF_GROUP_SIZE;
    }
    // the blocks started, then per block a flag and its total
    static vk::DeviceSize cdf_state_size(const uint32_t count) {
        return (1 + 3 * static_cast<vk::DeviceSize>(cdf_block_count(count))) * sizeof(uint32_t);
    }
    uint32_t setup_group_count() const {
        const uint32_t per_group = LIGHT_GRID_SETUP_GROUP * LIGHT_GRID_SETUP_ITEMS;
        return (triangle_count + per_group - 1) / per_group;
    }
    uint32_t update_lanes() const {
        const float ideal = 900.f / std::sqrt(static_cast<float>(std::max(triangle_count, 1u)));
        uint32_t lanes = 1;
        while (lanes < LIGHT_UPDATE_GROUP &&
               static_cast<float>(lanes) * std::numbers::sqrt2_v<float> < ideal)
            lanes *= 2;
        return lanes;
    }
    uint32_t sort_tile_count() const {
        return (triangle_count + LIGHT_SORT_TILE - 1) / LIGHT_SORT_TILE;
    }
    // per pass the digit counts, the tiles started and every tile's digit status
    vk::DeviceSize sort_state_size() const {
        return static_cast<vk::DeviceSize>(LIGHT_SORT_PASSES) *
               ((1u << LIGHT_SORT_RADIX_BITS) * (sort_tile_count() + 1) + 1) * sizeof(uint32_t);
    }
    uint32_t grid_cell_count() const {
        const uint32_t d = static_cast<uint32_t>(grid_dimension);
        return static_cast<uint32_t>(grid_cascades) * d * d * d;
    }
    void ensure_buffer(BufferHandle& buffer,
                       vk::DeviceSize size,
                       const std::string& name,
                       const CommandBufferHandle& cmd);
    static void retire_buffer(BufferHandle& buffer, const CommandBufferHandle& cmd);

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;

    bool enabled = true;
    bool pool_presampled = true;
    bool grid_enabled = true;
    int32_t env_selection = EnvSelection::EnvSelectionPool;
    int32_t env_pool_size = 8192;
    int32_t pool_size = 4096;
    int32_t scene_draws = 3;
    int32_t grid_dimension = 16;
    int32_t grid_cascades = 6;
    int32_t grid_refinements = 2;
    int32_t cell_draws = 2;
    float grid_cell_size = 0.f; // 0: derived from the distance to the lights
    float grid_coverage = 1.f;
    float grid_jitter = 1.f;
    bool debug_jitter = true;
    int32_t debug_view = LightDebugView::LightDebugDrawOutcome;
    bool constants_dirty = true;
    float grid_share = 1.f;
    float grid_even_share = 0.3f;
    int32_t slot_weighing = LightSlotWeighing::LightSlotWeighingCell;
    bool grid_lists_env = true;
    // set for a frame the carried-over grid cannot describe
    bool grid_reset = true;
    float3 camera_position{0.f};
    uint32_t frame = 0;
    bool env_share_from_power = true;
    float env_share = 0.5f;
    // Neither technique may lose its density where both can contribute.
    float env_share_min = 0.05f;
    float env_share_max = 0.95f;
    float scene_radius = 0.f;
    bool env_emissive = false;
    bool has_sky_portals = false;
    int32_t flux_samples = 32;
    // log2 side length of the equal-area octahedral importance map over the environment
    int32_t env_importance_log2 = 8;
    bool env_importance_resized = false;
    bool env_importance_built = false;
    uint64_t env_version = std::numeric_limits<uint64_t>::max();

    std::vector<LightGeometry> light_geometries;
    std::vector<uint32_t> geometry_light_offsets;
    uint32_t triangle_count = 0;
    bool tables_dirty = false;

    ShaderObjectAllocatorHandle fallback_obj_allocator;

    BufferHandle env_importance_built_buffer;
    BufferHandle env_pool_buffer;
    BufferHandle env_split_buffer;
    BufferHandle env_cdf_buffer;
    BufferHandle env_cdf_state_buffer;
    BufferHandle pool_buffer;
    BufferHandle env_cones_buffer;
    // side length of the environment the cones were computed for
    uint32_t env_cones_size = 0;
    // sorted into slot 0
    BufferHandle tree_keys_buffer[2];
    BufferHandle tree_values_buffer[2];
    BufferHandle tree_rank_buffer;
    BufferHandle tree_cdf_buffer;
    BufferHandle tree_cdf_state_buffer;
    BufferHandle tree_info_buffer[2];
    BufferHandle setup_state_buffer;
    BufferHandle sort_state_buffer;
    BufferHandle grid_keys_buffer[2];
    BufferHandle grid_contribution_buffer[2];
    BufferHandle grid_feedback_buffer[2];
    BufferHandle grid_touched_buffer[2];
    // zeroed in the next update
    bool grid_feedback_fresh = false;
    BufferHandle grid_starts_buffer;
    BufferHandle grid_estimate_buffer;
    BufferHandle grid_slot_bounds_buffer;
    BufferHandle grid_probability_buffer;
    BufferHandle grid_info_buffer[2];
    uint32_t grid_slot = 0;
    BufferHandle triangles_buffer;
    BufferHandle proxies_buffer;
    BufferHandle regions_buffer;
    BufferHandle light_geometries_buffer;
    BufferHandle geometry_light_offsets_buffer;

    SlangCompositionHandle update_composition;
    Versioned<SlangProgram> update_program;
    Versioned<SlangProgramEntryPoint> update_entry_point;
    Versioned<Pipeline> update_pipeline;
    Versioned<ShaderObject> update_params;

    SlangCompositionHandle preprocess_composition;
    Versioned<SlangProgram> preprocess_program;
    Versioned<SlangProgramEntryPoint> setup_entry_point;
    Versioned<SlangProgramEntryPoint> pool_entry_point;
    Versioned<SlangProgramEntryPoint> grid_entry_point;
    Versioned<SlangProgramEntryPoint> env_split_entry_point;
    Versioned<Pipeline> setup_pipeline;
    Versioned<Pipeline> pool_pipeline;
    Versioned<Pipeline> grid_pipeline;
    Versioned<Pipeline> env_split_pipeline;
    Versioned<ShaderObject> setup_params;
    Versioned<ShaderObject> pool_params;
    Versioned<SlangProgramEntryPoint> env_cones_entry_point;
    Versioned<Pipeline> env_cones_pipeline;
    Versioned<ShaderObject> env_cones_params;
    Versioned<ShaderObject> grid_params;
    Versioned<ShaderObject> env_split_params;

    SlangCompositionHandle sort_composition;
    Versioned<SlangProgram> sort_program;
    Versioned<SlangProgramEntryPoint> sort_histogram_entry_point;
    Versioned<SlangProgramEntryPoint> sort_scatter_entry_point;
    Versioned<Pipeline> sort_histogram_pipeline;
    Versioned<Pipeline> sort_scatter_pipeline;
    // the histogram, then one per pass
    std::vector<Versioned<ShaderObject>> sort_params;

    SlangCompositionHandle env_composition;
    Versioned<SlangProgram> env_program;
    Versioned<SlangProgramEntryPoint> env_pool_entry_point;
    Versioned<Pipeline> env_pool_pipeline;
    Versioned<ShaderObject> env_pool_params;
    Versioned<SlangProgramEntryPoint> env_build_entry_point;
    Versioned<Pipeline> env_build_pipeline;
    Versioned<ShaderObject> env_build_params;

    SlangCompositionHandle cdf_composition;
    Versioned<SlangProgram> cdf_program;
    Versioned<SlangProgramEntryPoint> cdf_entry_point;
    Versioned<Pipeline> cdf_pipeline;
    Versioned<ShaderObject> cdf_params;
};

} // namespace merian
