#pragma once

#include "merian-shaders/scene/scene-data.slangh"
#include "merian-shaders/shading/materials/material_system.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/camera/camera.hpp"
#include "merian/utils/versionable.hpp"
#include "merian/vk/raytrace/as_builder.hpp"

#include <optional>
#include <set>
#include <vector>

namespace merian {

enum class GeometryFlags : uint32_t {
    None = 0,
    IsOpaque = 0x1,
    IsDynamic = 0x2,
};

inline GeometryFlags operator|(GeometryFlags a, GeometryFlags b) {
    return static_cast<GeometryFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline bool operator&(GeometryFlags a, GeometryFlags b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

using NodeID = uint32_t;
using MeshID = uint32_t;
static constexpr NodeID NODE_ID_INVALID = UINT32_MAX;

struct Mesh {
    std::vector<PackedVertexData> vertices;
    std::vector<uint3> indices;

    MaterialID material_id;
    GeometryFlags flags = GeometryFlags::IsOpaque;

    // Set of scene graph nodes that instance this mesh.
    // Populated by add_mesh_instance. A mesh with >1 instance shares a BLAS.
    std::set<NodeID> instances;
};

// Scene graph node (transform hierarchy).
struct SceneNode {
    NodeID parent = NODE_ID_INVALID;

    std::string name;

    float4x4 local_transform = identity();
    float4x4 global_transform = identity();

    std::vector<NodeID> children;
};

// A group of meshes that share a BLAS.
// - Static non-instanced: all in one group, pre-transformed, TLAS identity.
// - Dynamic non-instanced: grouped by shared globalMatrixID.
// - Instanced: grouped by identical instance set (same NodeIDs).
struct MeshGroup {
    std::vector<MeshID> mesh_list;
    bool is_static = true;
};

class SceneError : std::runtime_error {
  public:
    SceneError(const std::string& msg) : std::runtime_error(msg) {}
};

class Scene : public Versionable, public std::enable_shared_from_this<Scene> {
  public:
    Scene(const ShaderCompileContextHandle& compile_context,
          const ContextHandle& context,
          const ResourceAllocatorHandle& allocator,
          const ShaderObjectAllocatorHandle& obj_allocator,
          const MaterialSystemHandle& material_system);

    virtual ~Scene() = default;

    void update(const CommandBufferHandle& cmd, float time, float time_diff, uint32_t frame);

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    bool set_build_acceleration_structure(bool build);

    bool get_build_acceleration_structure() const {
        return build_as;
    }

    const MaterialSystemHandle& get_material_system() const {
        return material_system;
    }

    const TextureManagerHandle& get_texture_manager() const {
        return material_system->get_texture_manager();
    }

    const ShaderObjectHandle& get_shader_object() const {
        return shader_object;
    }

    operator const ShaderObjectHandle&() const {
        return shader_object;
    }

    CameraHandle get_active_camera() const;
    void set_active_camera(uint32_t index);

    bool has_geometry() const {
        return !meshes.empty();
    }

    const std::vector<SceneNode>& get_scene_graph() const {
        return scene_graph;
    }

    // Bake single-instance dynamic mesh world transforms on CPU at upload time
    // (small scenes / debugging). Static meshes are always pre-transformed.
    bool get_pretransform_dynamic() const {
        return pretransform_dynamic;
    }
    void set_pretransform_dynamic(bool value);

    void properties(Properties& props);

  protected:
    virtual void on_update(float time, float time_diff) {
        (void)time;
        (void)time_diff;
    }

    MeshID add_mesh(Mesh mesh);
    NodeID add_node(SceneNode node);
    void add_mesh_instance(MeshID mesh_id, NodeID node_id);
    void add_camera(CameraHandle camera);
    void compute_world_transforms();

    std::vector<Mesh> meshes;
    std::vector<SceneNode> scene_graph;
    bool pretransform_dynamic = false;

  private:
    void update_composition_constants();
    void rebuild_shader_object();
    void upload_geometry_buffers(const CommandBufferHandle& cmd);
    void create_mesh_groups();
    void build_blas(const CommandBufferHandle& cmd);
    void build_tlas(const CommandBufferHandle& cmd);

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderObjectAllocatorHandle obj_allocator;
    MaterialSystemHandle material_system;

    std::vector<MeshGroup> mesh_groups;

    // Per-mesh: list of geometry instance indices for each instance of this mesh.
    // mesh_id_to_instance_ids[mesh_id][i] = global geometry instance index.
    std::vector<std::vector<uint32_t>> mesh_id_to_instance_ids;

    // Flat array of GeometryData ordered for InstanceID+GeometryIndex lookup.
    std::vector<GeometryData> geometry_instance_data;

    std::vector<BufferHandle> vertex_buffers;
    std::vector<BufferHandle> index_buffers;
    BufferHandle geometry_data_buffer;
    BufferHandle instance_transforms_buffer;
    BufferHandle inverse_transposed_instance_transforms_buffer;
    BufferHandle prev_instance_transforms_buffer;
    BufferHandle prev_inverse_transposed_instance_transforms_buffer;
    std::vector<float4x4> prev_instance_transforms_data;

    bool build_as = false;
    std::optional<ASBuilder> as_builder;
    std::vector<AccelerationStructureHandle> blas_list;
    AccelerationStructureHandle tlas;
    BufferHandle tlas_instances_buffer;
    BufferHandle scratch_buffer;
    bool bvh_dirty = true;

    std::vector<CameraHandle> cameras;
    uint32_t active_camera = 0;
    Camera prev_active_camera;
    bool geometry_dirty = true;

    SlangCompositionHandle composition;
    SlangProgramHandle layout_program;
    ShaderObjectHandle shader_object;
};

using SceneHandle = std::shared_ptr<Scene>;

} // namespace merian
