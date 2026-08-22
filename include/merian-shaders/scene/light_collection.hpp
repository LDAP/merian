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
    void update(const CommandBufferHandle& cmd,
                const SlangCompositionHandle& scene_composition,
                const ShaderObjectHandle& scene_object,
                const ShaderObjectAllocatorHandle& obj_allocator,
                uint32_t frame);

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

    // Probability of sampling the environment instead of a triangle (if both exist).
    float get_env_probability() const {
        return env_probability;
    }

    void set_env_probability(const float p) {
        env_probability = p;
    }

    // Whether the scene's environment map emits; false routes all samples to the triangles.
    void set_env_emissive(const bool value) {
        env_emissive = value;
    }

    // The grid follows the camera.
    void set_acceleration_structure(const AccelerationStructureHandle& as) {
        acceleration_structure = as;
    }

    void set_camera(const float3& position) {
        camera_position = position;
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
    uint32_t env_importance_quad_count() const;
    uint32_t grid_cell_count() const {
        const uint32_t d = static_cast<uint32_t>(grid_dimension);
        return d * d * d;
    }
    void ensure_buffer(BufferHandle& buffer,
                       vk::DeviceSize size,
                       const std::string& name,
                       const CommandBufferHandle& cmd);

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;

    bool enabled = true;
    int32_t selection = LightSelection::LightSelectionGrid;
    int32_t env_selection = EnvSelection::EnvSelectionPool;
    int32_t env_pool_size = 8192;
    int32_t pool_size = 4096;
    int32_t pool_tile_size = 256;
    int32_t grid_dimension = 32;
    int32_t grid_candidates = 8;
    float grid_probability = 0.7f;
    AccelerationStructureHandle acceleration_structure;
    float grid_cell_size = 0.f; // 0: derived from the light bounds
    float grid_coverage = 0.5f;
    bool grid_visibility = true;
    float3 camera_position{0.f};
    float env_probability = 0.5f;
    bool env_emissive = false;
    bool has_sky_portals = false;
    int32_t flux_samples = 32;
    // log2 side length of the equal-area octahedral importance map over the environment
    int32_t env_importance_log2 = 8;
    bool env_importance_resized = false;

    std::vector<LightGeometry> light_geometries;
    std::vector<uint32_t> geometry_light_offsets;
    uint32_t triangle_count = 0;
    bool tables_dirty = false;

    ShaderObjectAllocatorHandle fallback_obj_allocator;

    BufferHandle env_importance_buffer;
    BufferHandle env_pool_buffer;
    BufferHandle weights_buffer;
    BufferHandle pool_buffer;
    BufferHandle grid_buffer;
    BufferHandle grid_info_buffer;
    BufferHandle triangles_buffer;
    BufferHandle cdf_buffer;
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
    Versioned<SlangProgramEntryPoint> weights_entry_point;
    Versioned<SlangProgramEntryPoint> pool_entry_point;
    Versioned<SlangProgramEntryPoint> grid_entry_point;
    Versioned<Pipeline> setup_pipeline;
    Versioned<Pipeline> weights_pipeline;
    Versioned<Pipeline> pool_pipeline;
    Versioned<Pipeline> grid_pipeline;
    Versioned<ShaderObject> setup_params;
    Versioned<ShaderObject> weights_params;
    Versioned<ShaderObject> pool_params;
    Versioned<ShaderObject> grid_params;

    SlangCompositionHandle env_composition;
    Versioned<SlangProgram> env_program;
    Versioned<SlangProgramEntryPoint> env_pool_entry_point;
    Versioned<Pipeline> env_pool_pipeline;
    Versioned<ShaderObject> env_pool_params;
    Versioned<SlangProgramEntryPoint> env_build_entry_point;
    Versioned<SlangProgramEntryPoint> env_reduce_entry_point;
    Versioned<Pipeline> env_build_pipeline;
    Versioned<Pipeline> env_reduce_pipeline;
    Versioned<ShaderObject> env_build_params;
    std::vector<Versioned<ShaderObject>> env_reduce_params;

    SlangCompositionHandle cdf_composition;
    Versioned<SlangProgram> cdf_program;
    Versioned<SlangProgramEntryPoint> cdf_entry_point;
    Versioned<Pipeline> cdf_pipeline;
    Versioned<ShaderObject> cdf_params;
};

} // namespace merian
