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
using CameraID = uint32_t;

// Means this node is a root node and its local transform is the global transform.
static constexpr NodeID NODE_ID_INVALID = UINT32_MAX;
static constexpr CameraID CAMERA_ID_INVALID = UINT32_MAX;

struct SceneNode {
    std::string name;

    NodeID parent = NODE_ID_INVALID;
    std::vector<NodeID> children;

    float4x4 local_transform = identity();
    // Empty if invalidated; all children must be invalidated as well.
    std::optional<float4x4> global_transform;
    // Cached inverse-transpose of global_transform; computed alongside global_transform.
    std::optional<float4x4> global_inverse_transposed;
};

inline std::string format_as(const SceneNode& node) {
    return fmt::format(
        "name: {}\nparent: {}\nnum children: {}\nlocal_transform:\n{}\nglobal_transform:\n{}",
        node.name.empty() ? "<none>" : node.name, node.parent, node.children.size(),
        node.local_transform, node.global_transform.value_or(float4x4(0)));
}

class Mesh {
  public:
    std::string name;
    MaterialID material_id{};
    GeometryFlags flags = GeometryFlags::IsOpaque;

    // Populated by add_mesh_instance. A mesh with >1 instance shares a BLAS.
    std::set<NodeID> instances;

    virtual ~Mesh() = default;

    virtual uint32_t get_vertex_count() const = 0;
    virtual uint32_t get_primitive_count() const = 0;

    virtual float3 get_position(uint32_t vertex_idx) const = 0;
    virtual float3 get_normal(uint32_t vertex_idx) const = 0;
    virtual float2 get_uv(uint32_t vertex_idx) const = 0;
    // xyz = tangent direction, w = bitangent sign (+1 or -1)
    virtual float4 get_tangent(uint32_t vertex_idx) const = 0;

    virtual uint3 get_indices(uint32_t primitive_idx) const = 0;

    PackedVertexData get_packed_vertex(uint32_t vertex_idx) const;
    PackedVertexData get_packed_vertex_pretransformed(uint32_t vertex_idx,
                                                      const SceneNode& node) const;
};

using MeshHandle = std::unique_ptr<Mesh>;

inline std::string format_as(const Mesh& mesh) {
    return fmt::format("vertices: {}\ntriangles: {}\nmaterial id: {}\nnum instances: {}",
                       mesh.get_vertex_count(), mesh.get_primitive_count(), mesh.material_id,
                       mesh.instances.size());
}

// Concrete Mesh that owns its vertex/index data.
class SimpleMesh : public Mesh {
  public:
    std::vector<PackedVertexData> vertices;
    std::vector<uint3> indices;

    uint32_t get_vertex_count() const override {
        return static_cast<uint32_t>(vertices.size());
    }
    uint32_t get_primitive_count() const override {
        return static_cast<uint32_t>(indices.size());
    }

    float3 get_position(uint32_t vertex_idx) const override {
        return vertices[vertex_idx].position;
    }
    float3 get_normal(uint32_t vertex_idx) const override;
    float2 get_uv(uint32_t vertex_idx) const override {
        return float2(vertices[vertex_idx].uv);
    }
    float4 get_tangent(uint32_t vertex_idx) const override;

    uint3 get_indices(uint32_t primitive_idx) const override {
        return indices[primitive_idx];
    }
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

    // We use a BLAS / TLAS organization that is inspired by by Falcor.
    // https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Scene/Scene.h
    //
    // BLASs (one for each group)
    // - static, non-instanced -> pretransform and put in single group
    // - dynamic, non-instanced -> group if same transform
    // - instanced -> one BLAS for each group with identical instances
    // - procedural -> own group / BLAS
    //
    // TLAS
    // - We set InstanceID as the prefix sum of the number of geometries with lower InstanceIndex.
    // That means InstanceID + GeometryIndex is unique. We call that GeometryID.
    //
    // clang-format off
    //                           ----------------------------------------------------------------------------------------------
    //                           |                                         Value(s)                                           |
    //    ---------------------------------------------------------------------------------------------------------------------
    //    | InstanceID           |  0                    |  4  |  5  |  6  |  7  |  8  |  9     |  10          |  13          |
    //    | InstanceContribution |  0                    |  4  |  5  |  6  |  7  |  7  |  7     |  8           |  8           |
    //    | BLAS Geometry Index  |  0  ,  1  ,  2  ,  3  |  0  |  0  |  0  |  0                 |  0 , 1 , 2   |  0 , 1 , 2   |
    //    ---------------------------------------------------------------------------------------------------------------------
    //    | Notes                | Four geometries in    | One instance    | Multiple instances | Two instances of three      |
    //    |                      | one BLAS              | per geom/BLAS   | of same geom/BLAS  | geometries in one BLAS      |
    //    --------------------------------------------------------------------------------------------------------------------|
    // clang-format on

  public:
    Scene(const ShaderCompileContextHandle& compile_context,
          const ContextHandle& context,
          const ResourceAllocatorHandle& allocator,
          const ShaderObjectAllocatorHandle& obj_allocator,
          const MaterialSystemHandle& material_system);

    virtual ~Scene() = default;

    // ------------------------------

    void update(const CommandBufferHandle& cmd, float time, float time_diff, uint32_t frame);

    // ------------------------------

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

    std::vector<CameraHandle> get_cameras() const;

    CameraHandle get_camera(const CameraID camera_id) const;

    CameraHandle get_active_camera() const;

    void set_active_camera(uint32_t index);

    bool has_geometry() const {
        return !meshes.empty();
    }

    const std::vector<SceneNode>& get_scene_graph() const {
        return scene_graph;
    }

    const std::vector<MeshHandle>& get_meshes() const {
        return meshes;
    }

    const SceneNode& get_node(const NodeID node_id) {
        assert(node_id < scene_graph.size());
        return scene_graph[node_id];
    }

    // Guarantees that the global transform is available (unlike get_node).
    const float4x4& get_global_transform(const NodeID node_id) {
        assert(scene_graph[node_id].global_transform);
        return scene_graph[node_id].global_transform.value();
    }

    const Mesh& get_mesh(const MeshID mesh_id) {
        assert(mesh_id < meshes.size());
        return *meshes[mesh_id];
    }

    // Bake single-instance dynamic mesh world transforms on CPU at upload time
    // (small scenes / debugging). Static meshes are always pre-transformed.
    bool get_pretransform_dynamic() const {
        return pretransform_dynamic;
    }
    void set_pretransform_dynamic(bool value);

    // A the scenes up direction
    virtual float3 get_up() {
        return float3(0, 0, 1);
    }

    // ------------------------------

    void properties(Properties& props);

  protected:
    virtual void on_update(float time, float time_diff, uint32_t frame) {
        (void)time;
        (void)time_diff;
        (void)frame;
    }

    MeshID add_mesh(MeshHandle mesh);

    NodeID add_node(SceneNode node);

    void add_mesh_instance(MeshID mesh_id, NodeID node_id);

    CameraID add_camera(CameraHandle camera);

    // can be invalid if information is not available.
    AABB aabb;

  private:
    void node_properties(Properties& props, const SceneNode& node);
    void update_composition_constants();
    void rebuild_shader_object();
    void upload_geometry_buffers(const CommandBufferHandle& cmd);
    void create_mesh_groups();
    void build_blas(const CommandBufferHandle& cmd);
    void build_tlas(const CommandBufferHandle& cmd);

  private:
    // --------------------------
    // Context etc.

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderObjectAllocatorHandle obj_allocator;

    SlangCompositionHandle composition;
    SlangProgramHandle layout_program;
    ShaderObjectHandle shader_object;

    // --------------------------
    // Scene Definition

    MaterialSystemHandle material_system;
    std::vector<MeshHandle> meshes;
    std::vector<SceneNode> scene_graph;
    bool pretransform_dynamic = false;
    std::vector<MeshGroup> mesh_groups;
    std::vector<CameraHandle> cameras;
    uint32_t active_camera = 0;

    // --------------------------
    // Debug
    bool enable_debug_camera = false;
    CameraID debug_camera_id = CAMERA_ID_INVALID;

    // --------------------------
    // Cached and Precomputed

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
    Camera prev_active_camera;
    bool geometry_dirty = true;
};

using SceneHandle = std::shared_ptr<Scene>;

} // namespace merian
