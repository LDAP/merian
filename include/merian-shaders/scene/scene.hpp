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
#include <vector>

namespace merian {

enum GeometryFlags : uint32_t {
    None = 0,
    IsDynamic = 0x1,             // default: static
    IsOpaque = 0x2,              // default: treat all as non-opaque (allow alpha mask)
    FrontCounterClockwise = 0x4, // default: clockwise
    TwoSided = 0x8,              // default: cull backfaces
};

constexpr GeometryFlags operator|(GeometryFlags a, GeometryFlags b) {
    return static_cast<GeometryFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr bool operator&(GeometryFlags a, GeometryFlags b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

using NodeID = uint32_t;
using MeshID = uint32_t;
using CameraID = uint32_t;

// Means this node is a root node and its local transform is the global transform.
static constexpr NodeID NODE_ID_INVALID = UINT32_MAX;
static constexpr CameraID CAMERA_ID_INVALID = UINT32_MAX;

class Node {
  public:
    std::string name;

    NodeID parent = NODE_ID_INVALID;
    std::vector<NodeID> children;

    float4x4 local_transform = identity();

    // ------------------
    // Managed by Scene

    // Empty if invalidated; all children must be invalidated as well.
    std::optional<float4x4> global_transform;
    // Cached inverse-transpose of global_transform; computed alongside global_transform.
    std::optional<float4x4> global_inverse_transposed;
};

inline std::string format_as(const Node& node) {
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
    bool dirty = true;

    virtual ~Mesh() = default;

    virtual uint32_t get_vertex_count() const = 0;
    virtual uint32_t get_primitive_count() const = 0;

    virtual float3 get_position(uint32_t vertex_idx) const = 0;
    virtual float3 get_prev_position(uint32_t vertex_idx) const {
        assert((flags & GeometryFlags::IsDynamic) == 0);
        return get_position(vertex_idx);
    }
    virtual float3 get_normal(uint32_t vertex_idx) const = 0;
    virtual float2 get_uv(uint32_t vertex_idx) const = 0;
    // bitangent = cross(normal, tangent.xyz) * tangent.w
    virtual float4 get_tangent(uint32_t vertex_idx) const = 0;

    virtual uint3 get_indices(uint32_t primitive_idx) const = 0;

    // ------------------------
    // default implementation calls the above methods
    // can be overwritten for performance

    virtual PackedVertexData get_packed_vertex(uint32_t vertex_idx) const;
    virtual PackedVertexData get_packed_vertex_pretransformed(uint32_t vertex_idx,
                                                              const Node& node) const;
    virtual PackedPrevVertexData get_packed_prev_vertex(uint32_t vertex_idx) const;
    virtual PackedPrevVertexData get_packed_prev_vertex_pretransformed(uint32_t vertex_idx,
                                                                       const Node& node) const;

    bool is_dynamic() const {
        return flags & GeometryFlags::IsDynamic;
    }

    bool is_static() const {
        return !(flags & GeometryFlags::IsDynamic);
    }

    bool is_front_counterclockwise() const {
        // Vulkan default is clockwise
        return flags & GeometryFlags::FrontCounterClockwise;
    }

    bool is_two_sided() const {
        // if yes, needs to disable backface culling
        return flags & GeometryFlags::TwoSided;
    }

    bool is_opaque() const {
        // if yes, allows to set the force opaque flag when raytracing.
        return flags & GeometryFlags::IsOpaque;
    }

    // ------------------
    // Managed by Scene

    std::unordered_set<NodeID> instances;
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

class SceneError : std::runtime_error {
  public:
    SceneError(const std::string& msg) : std::runtime_error(msg) {}
};

class Scene : public Versionable, public std::enable_shared_from_this<Scene> {
  private:
    // --------------------------
    // Internal types

    using MeshGroupID = uint32_t;

    struct MeshGroup {
        std::unordered_set<MeshID> meshes;

        // flags shared by all meshes
        GeometryFlags flags;

        // ----------------
        AccelerationStructureHandle blas;
        // a mesh changed (new mesh in group -> not the same group, never true)
        bool blas_dirty = false;

        // ----------------

        const std::unordered_set<NodeID>&
        get_instances(const std::vector<MeshHandle>& meshes) const {
            assert(!meshes.empty());
            return meshes[*this->meshes.begin()]->instances;
        }

        bool is_pretranformed(const std::vector<MeshHandle>& meshes,
                              bool pretransform_dynamic) const {
            assert(!meshes.empty());
            const Mesh& mesh = *meshes[*this->meshes.begin()];
            return (mesh.is_static() || pretransform_dynamic) && mesh.instances.size() <= 1;
        }
    };

  public:
    // We use a BLAS / TLAS organization that is inspired by by Falcor.
    // https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Scene/Scene.h
    //
    // BLASs (one for each group)
    // - static, non-instanced -> pretransform and put in a group
    // - dynamic, non-instanced -> group if same transform (same NodeID) and same flags
    // - instanced -> group if identical instances and flags (instanced cannot be
    // pretransformed!)
    // - procedural -> own group / BLAS
    //
    // - Groups must be further split if they differ in these GeometryFlags: ForceOpaque,
    // FrontCounterClockwise and TwoSided as those can only be set on the (BLAS) instance, not the
    // geometry.
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

    // Enable once Slang link time types work
    // bool set_build_acceleration_structure(bool build);

    // bool get_build_acceleration_structure() const {
    //     return build_as;
    // }

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

    const std::vector<Node>& get_scene_graph() const {
        return scene_graph;
    }

    const std::vector<MeshHandle>& get_meshes() const {
        return meshes;
    }

    const Node& get_node(const NodeID node_id) {
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

    // Scene-time hook. The default returns the update time unchanged. Scenes
    // that own a separate clock can override.
    virtual float get_time(float time) {
        return time;
    }

    // ------------------------------

    void properties(Properties& props);

    //
    const AABB& get_aabb() const {
        return aabb;
    }

  protected:
    virtual void
    on_update(const CommandBufferHandle& cmd, float time, float time_diff, uint32_t frame) {
        (void)cmd;
        (void)time;
        (void)time_diff;
        (void)frame;
    }

    MeshID add_mesh(MeshHandle mesh);

    // signalize that the geometry of this mesh has changed and needs to be reuploaded to the GPU
    // and BVHs with this mesh need to be rebuilt.
    void mark_mesh_dirty(MeshID mesh_id);

    NodeID add_node(Node node);

    void add_mesh_instance(MeshID mesh_id, NodeID node_id);

    CameraID add_camera(CameraHandle camera);

    const ContextHandle& get_context() const {
        return context;
    }

    const ResourceAllocatorHandle& get_allocator() const {
        return allocator;
    }

    // can be invalid if information is not available.
    AABB& get_aabb() {
        return aabb;
    }

  private:
    void ensure_index_vertex_buffers(const std::size_t min_index_buffer_count,
                                     const std::size_t min_vertex_buffer_count);
    void rebuild_shader_object();

    // Computes mesh groups according to the grouping logic above.
    // All groups share the same instances (transforms) -> iterating over groups and instances gives
    // the instances One group builds one BLAS -> the instances can refer to those.
    void compute_mesh_groups();

    // uploads the meshes, geometry data, and instance transforms
    void upload_geometry_buffers(const CommandBufferHandle& cmd);

    void build_blas(const CommandBufferHandle& cmd);
    void build_tlas(const CommandBufferHandle& cmd);

    void node_properties(Properties& props, const Node& node);

  private:
    static const std::size_t INITIAL_INDEX_BUFFER_COUNT = 128;
    static const std::size_t INITIAL_VERTEX_BUFFER_COUNT = 128;

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
    // CPU side scene definition

    MaterialSystemHandle material_system;
    std::vector<MeshHandle> meshes;
    std::vector<Node> scene_graph;
    bool pretransform_dynamic = false;
    std::vector<CameraHandle> cameras;
    uint32_t active_camera = 0;
    AABB aabb; // can be invalid if information is not available.

    // --------------------------
    // GPU side scene definition

    AccelerationStructureHandle tlas;

    // Indexed with GeometryID (InstanceID + GeometryIndex)
    BufferHandle geometries_buffer;

    // Indexed with MeshID / geometry.index_buffer_index -> PrimitiveID
    std::vector<BufferHandle> index_buffers;
    // Indexed with MeshID / geometry.vertex_buffer_index -> indices
    std::vector<BufferHandle> vertex_buffers;
    // Indexed with MeshID / geometry.vertex_buffer_index -> indices
    std::vector<BufferHandle> prev_vertex_buffers;

    // --------------------------
    // Debug
    bool enable_debug_camera = false;
    CameraID debug_camera_id = CAMERA_ID_INVALID;

    // --------------------------
    // Cached and Precomputed

    bool needs_regroup = false; // a mesh was instanced

    static const MeshGroupID MESH_GROUP_ID_INVALID = MeshGroupID(-1);
    std::vector<MeshGroup> mesh_groups;
    // MeshID -> GroupID (MESH_GROUP_ID_INVALID if not in group, eg. because there was no instance
    // of the mesh)
    std::vector<MeshGroupID> mesh_to_group;

    // // Per-mesh: list of geometry instance indices for each instance of this mesh.
    // // mesh_id_to_instance_ids[mesh_id][i] = global geometry instance index.
    // std::vector<std::vector<uint32_t>> mesh_id_to_instance_ids;

    // // Per-mesh: index of the mesh group containing this mesh. UINT32_MAX if
    // // the mesh is not assigned to any group (no instances). Set by
    // // create_mesh_groups; consumed by mark_mesh_data_dirty handling.
    // std::vector<uint32_t> mesh_id_to_group_id;

    // // Per-mesh dirty bit: when true, the mesh's vertex/index data must be
    // // re-uploaded and its group's BLAS rebuilt on the next update(). Set
    // // via mark_mesh_data_dirty; cleared inside update().
    // std::vector<bool> mesh_data_dirty;

    // // Flat array of GeometryData ordered for InstanceID+GeometryIndex lookup.
    // std::vector<GeometryData> geometry_instance_data;
    // std::vector<BufferHandle> vertex_buffers;
    // std::vector<BufferHandle> index_buffers;
    // BufferHandle geometry_data_buffer;
    // BufferHandle instance_transforms_buffer;
    // BufferHandle inverse_transposed_instance_transforms_buffer;
    // BufferHandle prev_instance_transforms_buffer;
    // BufferHandle prev_inverse_transposed_instance_transforms_buffer;
    // std::vector<float4x4> prev_instance_transforms_data;

    // bool build_as = false;
    // std::optional<ASBuilder> as_builder;
    // std::vector<AccelerationStructureHandle> blas_list;
    // AccelerationStructureHandle tlas;
    // BufferHandle tlas_instances_buffer;
    // // Keepalive ring for tlas_instances_buffer: build_tlas can be called every
    // // frame for dynamic scenes, and the previous buffer may still be referenced
    // // by an in-flight command buffer. Sizing is conservative (covers up to 4
    // // frames in flight, which is more than any current swapchain config uses).
    // static constexpr uint32_t TLAS_INSTANCES_KEEPALIVE = 4;
    // std::array<BufferHandle, TLAS_INSTANCES_KEEPALIVE> tlas_instances_keepalive;
    // uint32_t tlas_instances_keepalive_idx = 0;
    // BufferHandle scratch_buffer;
    // bool bvh_dirty = true;
    // Camera prev_active_camera;
    // bool geometry_dirty = true;

    // // Placeholder fallback for the empty-scene case: lets us bind a real
    // // (but empty) buffer/TLAS to every Scene descriptor slot when no
    // // geometry has been added yet, so consumers that gate on
    // // has_geometry() (e.g. gbuffer_rt) can still construct/validate their
    // // descriptor sets.
    // //
    // // TODO: with VK_EXT_robustness2's nullDescriptor enabled the
    // // bindings can legally stay null and this whole fallback can be
    // // skipped. Wire a fast path once the device feature is requested.
    // BufferHandle placeholder_buffer;
    // AccelerationStructureHandle placeholder_tlas;
};

using SceneHandle = std::shared_ptr<Scene>;

} // namespace merian
