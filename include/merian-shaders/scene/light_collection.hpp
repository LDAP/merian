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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

namespace merian {

class Properties;

// The emissive triangles of a Scene and the light cuts NEE draws from.
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

    // prev_remap maps the previous frame's GeometryIDs to this frame's, empty where kept
    void set_geometries(const std::vector<EmissiveGeometry>& geometries,
                        uint32_t geometry_count,
                        const std::vector<GeometryRemap>& prev_remap);

    // call before write_to
    void prepare(const CommandBufferHandle& cmd);

    // Records the update passes and binds the scene object, which must not be written afterwards.
    void update_constants(const SlangCompositionHandle& scene_composition);

    void update(const CommandBufferHandle& cmd,
                const SlangCompositionHandle& scene_composition,
                const ShaderObjectHandle& scene_object,
                const ShaderObjectAllocatorHandle& obj_allocator);

    void write_to(ShaderCursor cursor) const;

    // looked up by this frame's GeometryIDs; this frame's densities where unknown
    void write_prev_to(ShaderCursor cursor) const;

    // Forgets what the cuts learned and the previous frame.
    void start_run();

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

    bool densities_vary() const {
        return enabled && triangle_count > 0 && (feedback || uses_grid());
    }

    void set_enabled(const bool value) {
        enabled = value;
    }

    void set_env_state(const uint64_t version, const bool is_static) {
        if (version != env_version || !is_static) {
            env_importance_built = false;
        }
        if (is_static != env_static && env_importance_log2 != env_dynamic_importance_log2) {
            env_importance_resized = true;
            grid_reset = true;
        }
        env_version = version;
        env_static = is_static;
    }

    void set_env_emissive(const bool value) {
        env_emissive = value;
    }

    // The grid follows the camera.
    void set_camera(const float3& position) {
        camera_position = position;
    }

    // UseEnvMap geometry exists
    void set_has_sky_portals(const bool value) {
        has_sky_portals = value;
    }

  private:
    struct Binding {
        BufferHandle triangles;
        BufferHandle regions;
        BufferHandle geometries;
        BufferHandle geometry_offsets;
        uint32_t triangle_count = 0;
        uint32_t geometry_count = 0;
        uint32_t geometry_offset_count = 0;
        BufferHandle cdf;
        BufferHandle guide;
        BufferHandle order;
        BufferHandle rank;
        uint32_t env_texels = 0;
        BufferHandle starts;
        BufferHandle env_prefix;
        BufferHandle cascades;
        BufferHandle index;
        BufferHandle blocks;
        bool in_use = false;
        float jitter = 0.f;
        float even_share = 0.f;
        BufferHandle contribution;
        BufferHandle estimate;
        BufferHandle slot_bounds;
        BufferHandle probability;
        BufferHandle feedback;
        uint32_t light_count = 0;
        bool has_sky_portals = false;
        uint32_t debug_view = 0;
        BufferHandle env_keys;
        BufferHandle env_cdf;
        BufferHandle env_guide;
        BufferHandle env_lookup;
        uint32_t env_size = 0;
        uint32_t env_buckets = 0;
    };

    void prepare_buffers(const CommandBufferHandle& cmd);
    Binding make_binding() const;
    static void write(ShaderCursor cursor, const Binding& binding);
    void ensure_pipelines(const SlangCompositionHandle& scene_composition);
    uint32_t env_importance_side_log2() const;
    uint32_t env_importance_size() const {
        return 1u << env_importance_side_log2();
    }
    uint32_t env_grid_cells(uint32_t grid_log2) const;
    uint32_t env_leaf_capacity() const;
    bool uses_grid() const {
        return source == LightSource::LightSourceGrid;
    }
    bool lists_env() const {
        return triangle_count > 0 && env_emissive && env_cdf_buffer[env_cdf_slot];
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
    // per bucket the first entry reaching it, then the entry count and the scale
    static vk::DeviceSize guide_size(const uint32_t buckets) {
        return (static_cast<vk::DeviceSize>(buckets) + 2) * sizeof(uint32_t);
    }
    uint32_t setup_group_count() const {
        const uint32_t per_group = LIGHT_GRID_SETUP_GROUP * LIGHT_GRID_SETUP_ITEMS;
        return (triangle_count + per_group - 1) / per_group;
    }
    uint32_t update_lanes() const {
        const float ideal = 900.f / std::sqrt(static_cast<float>(std::max(triangle_count, 1u)));
        const uint32_t max_lanes = std::min(LIGHT_UPDATE_GROUP, min_subgroup_size);
        uint32_t lanes = 1;
        while (lanes < max_lanes &&
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
    uint32_t cell_count() const {
        const uint32_t d = static_cast<uint32_t>(grid_dimension);
        return uses_grid()
                   ? LIGHT_GRID_ORIENTATIONS * static_cast<uint32_t>(grid_cascades) * d * d * d
                   : 1u;
    }
    uint32_t allocate_group_count() const {
        return (cell_count() + LIGHT_GRID_ALLOCATE_GROUP - 1) / LIGHT_GRID_ALLOCATE_GROUP;
    }
    uint32_t block_count() const {
        return uses_grid() ? static_cast<uint32_t>(grid_cells_kept) + 1u : 1u;
    }
    void ensure_buffer(BufferHandle& buffer,
                       vk::DeviceSize size,
                       const std::string& name,
                       const CommandBufferHandle& cmd);
    static void retire(BufferHandle& buffer, const CommandBufferHandle& cmd);

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;

    uint32_t min_subgroup_size;
    bool enabled = true;
    int32_t source = LightSource::LightSourceGrid;
    int32_t draws = 4;
    int32_t grid_dimension = 16;
    int32_t grid_cascades = 6;
    int32_t grid_cells_kept = 16384;
    int32_t grid_refinements = 2;
    float grid_cell_size = 0.f; // 0: derived from the distance to the lights
    float grid_coverage = 1.f;
    float grid_jitter = 1.f;
    bool debug_jitter = true;
    bool feedback = true;
    int32_t debug_view = LightDebugView::LightDebugDrawOutcome;
    bool constants_dirty = true;
    float grid_even_share = 0.3f;
    float triangle_prior = 0.5f;
    float env_prior = 0.005f;
    int32_t slot_weighing = LightSlotWeighing::LightSlotWeighingCell;
    // set for a frame the carried-over grid cannot describe
    bool grid_reset = true;
    float3 camera_position{0.f};
    uint32_t allocated_cells = 0;
    uint32_t allocated_blocks = 0;
    bool env_emissive = false;
    bool has_sky_portals = false;
    int32_t flux_samples = 32;
    // log2 side length of the equal-area octahedral importance map over the environment
    int32_t env_importance_log2 = 11;
    int32_t env_dynamic_importance_log2 = 9;
    int32_t env_importance_samples = 2;
    int32_t env_importance_leaves = 16384;
    bool env_importance_resized = false;
    bool env_importance_built = false;
    bool env_static = true;
    uint64_t env_version = std::numeric_limits<uint64_t>::max();

    std::vector<LightGeometry> light_geometries;
    std::vector<uint32_t> geometry_light_offsets;
    // per GeometryID of this frame the previous frame's first triangle
    std::vector<uint32_t> prev_light_offsets;
    bool prev_offsets_dirty = false;
    uint32_t triangle_count = 0;
    bool tables_dirty = false;

    // each buffer written only in its own frame
    std::optional<Binding> binding;
    std::optional<Binding> prev_binding;
    BufferHandle prev_offsets_buffer;

    BufferHandle env_texel_cdf_buffer;
    BufferHandle env_keys_buffer[2];
    BufferHandle env_cdf_buffer[2];
    BufferHandle env_guide_buffer[2];
    BufferHandle env_lookup_buffer[2];
    uint32_t env_cdf_slot = 0;
    BufferHandle env_cdf_state_buffer;
    BufferHandle env_leaves_state_buffer;
    BufferHandle env_cones_buffer;
    // side length of the environment the cones were computed for
    uint32_t env_cones_size = 0;
    // sorted into slot 0
    BufferHandle tree_keys_buffer[2];
    BufferHandle tree_values_buffer[2];
    BufferHandle tree_order_buffer[2];
    BufferHandle tree_rank_buffer[2];
    BufferHandle tree_cdf_buffer[2];
    BufferHandle tree_cdf_state_buffer;
    BufferHandle tree_guide_buffer[2];
    BufferHandle tree_info_buffer[2];
    // zeroed in the next update
    bool tree_info_fresh = false;
    BufferHandle setup_state_buffer;
    BufferHandle sort_state_buffer;
    BufferHandle grid_keys_buffer[2];
    BufferHandle grid_contribution_buffer[2];
    BufferHandle grid_feedback_buffer[2];
    // zeroed in the next update
    bool grid_feedback_fresh = false;
    BufferHandle grid_starts_buffer[2];
    BufferHandle grid_env_prefix_buffer[2];
    BufferHandle grid_estimate_buffer[2];
    BufferHandle grid_slot_bounds_buffer[2];
    BufferHandle grid_probability_buffer[2];
    BufferHandle grid_info_buffer[2];
    BufferHandle grid_index_buffer[2];
    BufferHandle grid_blocks_buffer;
    BufferHandle grid_free_buffer;
    BufferHandle grid_offsets_buffer;
    uint32_t grid_slot = 0;
    BufferHandle triangles_buffer[2];
    BufferHandle proxies_buffer;
    BufferHandle regions_buffer[2];
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
    Versioned<SlangProgramEntryPoint> grid_entry_point;
    Versioned<SlangProgramEntryPoint> keep_entry_point;
    Versioned<SlangProgramEntryPoint> allocate_entry_point;
    Versioned<SlangProgramEntryPoint> assign_entry_point;
    Versioned<Pipeline> setup_pipeline;
    Versioned<Pipeline> grid_pipeline;
    Versioned<Pipeline> keep_pipeline;
    Versioned<Pipeline> allocate_pipeline;
    Versioned<Pipeline> assign_pipeline;
    Versioned<ShaderObject> setup_params;
    Versioned<ShaderObject> keep_params;
    Versioned<ShaderObject> allocate_params;
    Versioned<ShaderObject> assign_params;
    Versioned<SlangProgramEntryPoint> env_cones_entry_point;
    Versioned<Pipeline> env_cones_pipeline;
    Versioned<ShaderObject> env_cones_params;
    Versioned<ShaderObject> grid_params;

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
    Versioned<SlangProgramEntryPoint> env_build_entry_point;
    Versioned<Pipeline> env_build_pipeline;
    Versioned<ShaderObject> env_build_params;
    Versioned<SlangProgramEntryPoint> env_leaves_entry_point;
    Versioned<Pipeline> env_leaves_pipeline;
    Versioned<ShaderObject> env_leaves_params;
    Versioned<SlangProgramEntryPoint> env_guide_entry_point;
    Versioned<Pipeline> env_guide_pipeline;
    Versioned<ShaderObject> env_guide_params;

    SlangCompositionHandle cdf_composition;
    Versioned<SlangProgram> cdf_program;
    Versioned<SlangProgramEntryPoint> cdf_entry_point;
    Versioned<Pipeline> cdf_pipeline;
    Versioned<ShaderObject> cdf_params;
    Versioned<SlangProgramEntryPoint> guide_entry_point;
    Versioned<Pipeline> guide_pipeline;
    Versioned<ShaderObject> guide_params;
};

} // namespace merian
