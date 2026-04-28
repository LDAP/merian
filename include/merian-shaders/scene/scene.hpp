#pragma once

#include "merian-shaders/scene/scene-data.slangh"
#include "merian-shaders/shading/materials/material_system.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/camera/camera.hpp"
#include "merian/utils/free_list.hpp"
#include "merian/utils/versionable.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/raytrace/as_builder.hpp"

#include <optional>
#include <variant>
#include <vector>

namespace merian {

enum class MeshFlags : uint32_t {
    None = 0,
    // default: not; means: vertices can change over time, get_prev_vertices must be valid.
    IsMorphed = 0x1,
    // default: treat all as non-opaque (allow alpha mask)
    IsOpaque = 0x2,
    // default: clockwise
    FrontCounterClockwise = 0x4,
    // default: cull backfaces
    TwoSided = 0x8,
};

constexpr MeshFlags operator|(MeshFlags a, MeshFlags b) {
    return static_cast<MeshFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr bool operator&(MeshFlags a, MeshFlags b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

inline std::string format_as(const MeshFlags flags) {
    if (flags == MeshFlags::None) {
        return "None";
    }
    std::string out;
    const auto append = [&](const char* name) {
        if (!out.empty()) {
            out += " | ";
        }
        out += name;
    };
    if (flags & MeshFlags::IsMorphed) {
        append("IsMorphed");
    }
    if (flags & MeshFlags::IsOpaque) {
        append("IsOpaque");
    }
    if (flags & MeshFlags::FrontCounterClockwise) {
        append("FrontCounterClockwise");
    }
    if (flags & MeshFlags::TwoSided) {
        append("TwoSided");
    }
    return out;
}

using NodeID = uint32_t;
using MeshID = uint32_t;
using CameraID = uint32_t;

// Means this node is a root node and its local transform is the global transform.
static constexpr NodeID NODE_ID_INVALID = UINT32_MAX;
static constexpr CameraID CAMERA_ID_INVALID = UINT32_MAX;

class SceneNode {
  public:
    std::string name;

    // Transform changes over time; update_node() asserts this. Set before add_node().
    bool is_animated = false;

    NodeID parent = NODE_ID_INVALID;
    std::vector<NodeID> children;

    float4x4 local_transform = identity();

    // ------------------
    // Managed by Scene

    // Empty if invalidated; all children must be invalidated as well.
    std::optional<float4x4> global_transform;
    // Cached inverse-transpose of global_transform; computed alongside global_transform.
    std::optional<float4x4> global_inverse_transposed;

    std::optional<float4x4> prev_global_transform;
    std::optional<float4x4> prev_global_inverse_transposed;
};

inline std::string format_as(const SceneNode& node) {
    return fmt::format(
        "name: {}\nparent: {}\nnum children: {}\nlocal_transform:\n{}\nglobal_transform:\n{}",
        node.name.empty() ? "<none>" : node.name, node.parent, node.children.size(),
        node.local_transform, node.global_transform.value_or(float4x4(0)));
}

class SceneNode;

class Mesh {
  public:
    // ----- Bulk CPU vertex data -----
    template <typename T> struct HostVertexSource {
        virtual ~HostVertexSource() = default;
        virtual void write(T* dst) const = 0;
        virtual void write_pretransformed(const float4x4& global_transform,
                                          const float4x4& global_inverse_transposed,
                                          T* dst) const = 0;
    };

    using HostVertices = HostVertexSource<PackedVertexData>*;
    using HostPrevVertices = HostVertexSource<PackedPrevVertexData>*;

    // ----- Bulk CPU index data -----
    struct HostIndexSource {
        virtual ~HostIndexSource() = default;
        virtual void write(void* dst) const = 0;
    };

    using HostIndices = HostIndexSource*;

    // ----- Static prepacked CPU Data -----
    template <typename T> struct HostPacked {
        const T* data;
    };

    // ----- GPU Staged Data (Frequently updated CPU data) -----
    struct DeviceStaged {
        BufferHandle data;
        vk::DeviceSize offset = 0;
    };

    // ----- GPU Local Data (Procedural/Static GPU data) -----
    struct DeviceLocal {
        BufferHandle data;
    };

    using MeshVertexData =
        std::variant<HostVertices, HostPacked<PackedVertexData>, DeviceStaged, DeviceLocal>;

    // can be std::monostate for static meshes.
    using MeshPrevVertexData = std::variant<std::monostate,
                                            HostPrevVertices,
                                            HostPacked<PackedPrevVertexData>,
                                            DeviceStaged,
                                            DeviceLocal>;

    // Layout depends on Mesh::index_type
    using MeshIndexData = std::variant<HostIndices, HostPacked<void>, DeviceStaged, DeviceLocal>;

  public:
    std::string name;
    MaterialID material_id{};
    MeshFlags flags = MeshFlags::IsOpaque;
    vk::IndexType index_type = vk::IndexType::eUint32;

    // different meanings depending on data type:
    // - Host*: reupload (+ pretransform) + device copy + TLAS build
    // - DeviceStaged: device copy (TODO: including pretransform on GPU) + TLAS build
    // - DeviceLocal: (TODO: pretransform on GPU) + TLAS build

    bool vertices_dirty = true;
    bool indices_dirty = true;

    virtual ~Mesh() = default;

    virtual uint32_t get_vertex_count() const = 0;
    virtual uint32_t get_primitive_count() const = 0;

    virtual MeshVertexData get_vertices() const = 0;
    virtual MeshPrevVertexData get_prev_vertices() const = 0;
    virtual MeshIndexData get_indices() const = 0;

    bool is_morphed() const {
        return flags & MeshFlags::IsMorphed;
    }

    bool is_dynamic() const {
        return is_morphed() || animated_instance_count > 0;
    }

    bool is_static() const {
        return !is_dynamic();
    }

    bool is_front_counterclockwise() const {
        // Vulkan default is clockwise
        return flags & MeshFlags::FrontCounterClockwise;
    }

    bool is_two_sided() const {
        // if yes, needs to disable backface culling
        return flags & MeshFlags::TwoSided;
    }

    bool is_opaque() const {
        // if yes, allows to set the force opaque flag when raytracing.
        return flags & MeshFlags::IsOpaque;
    }

    bool is_dirty() const {
        return vertices_dirty || indices_dirty;
    }

    // ------------------
    // Managed by Scene

    std::set<NodeID> instances;
    uint32_t animated_instance_count = 0;
};

using MeshHandle = std::unique_ptr<Mesh>;

inline std::string format_as(const Mesh& mesh) {
    return fmt::format(
        "vertices: {}\ntriangles: {}\nmaterial id: {}\nnum instances: {}\nanimated instances: "
        "{}\nflags: {}\nindex type: {}\nvertices dirty: {}\nindices dirty: {}",
        mesh.get_vertex_count(), mesh.get_primitive_count(), mesh.material_id,
        mesh.instances.size(), mesh.animated_instance_count, mesh.flags,
        vk::to_string(mesh.index_type), mesh.vertices_dirty, mesh.indices_dirty);
}

class SimpleMesh : public Mesh {
  public:
    std::vector<PackedVertexData> vertices;
    std::vector<PackedPrevVertexData> prev_vertices;
    std::vector<uint3> indices;

    uint32_t get_vertex_count() const override {
        return static_cast<uint32_t>(vertices.size());
    }
    uint32_t get_primitive_count() const override {
        return static_cast<uint32_t>(indices.size());
    }

    virtual MeshVertexData get_vertices() const override {
        return HostPacked<PackedVertexData>(vertices.data());
    }
    virtual MeshPrevVertexData get_prev_vertices() const override {
        if (!is_morphed()) {
            return std::monostate();
        }
        assert(prev_vertices.size() == vertices.size());
        HostPacked<PackedPrevVertexData> data(prev_vertices.data());
        return data;
    }
    virtual MeshIndexData get_indices() const override {
        assert(index_type == vk::IndexType::eUint32);
        return HostPacked<void>(indices.data());
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
        MeshFlags flags;
        // true if any instance node has is_animated set
        bool has_animated_node = false;

        // ----------------
        AccelerationStructureHandle blas;
        // a mesh changed (new mesh in group -> not the same group, never true)
        bool blas_dirty = false;

        // ----------------

        const std::set<NodeID>& get_instances(const std::vector<MeshHandle>& meshes) const {
            assert(!meshes.empty());
            return meshes[*this->meshes.begin()]->instances;
        }

        // means the Scene uploads the meshes in this group pretransformed and the instance
        // transform should be the identity.
        bool is_pretranformed(const std::vector<MeshHandle>& meshes,
                              bool pretransform_dynamic) const {
            assert(!meshes.empty());
            const Mesh& mesh = *meshes[*this->meshes.begin()];
            assert(flags == mesh.flags);
            if (mesh.instances.size() > 1)
                return false;
            if (has_animated_node && !pretransform_dynamic)
                return false;
            if (!mesh.is_morphed())
                return true;
            return pretransform_dynamic;
        }
    };

  public:
    // We use a BLAS / TLAS organization that is inspired by by Falcor.
    // https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Scene/Scene.h
    //
    // We differentiate between meshes with IsMorphed (vertices that change) and nodes with
    // IsAnimated (transforms that change). A mesh is "dynamic" if either applies to any of
    // its instances; otherwise it is "static".
    //
    // BLASs (one for each group, grouped by MeshFlags)
    // - non-animated, non-instanced -> pretransform and merge into one BLAS per MeshFlags
    //   (unless IsMorphed and pretransform_dynamic is off)
    // - animated, non-instanced     -> group by (MeshFlags, NodeID), not pretransformed
    // - instanced                   -> group by (instance set, MeshFlags), never pretransformed
    //
    // Groups must be split if they differ in IsOpaque, FrontCounterClockwise or TwoSided
    // since those map to per-TLAS-instance flags.
    //
    // TLAS
    // - InstanceID = prefix sum of geometry counts with lower InstanceIndex.
    //   InstanceID + GeometryIndex is unique and called GeometryID.
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

    const std::vector<std::optional<SceneNode>>& get_scene_graph() const {
        return scene_graph;
    }

    const std::vector<MeshHandle>& get_meshes() const {
        return meshes;
    }

    // to get transforms use get_*_transform(...)
    const SceneNode& get_node(const NodeID node_id) {
        assert(node_ids.is_used(node_id));
        return *scene_graph[node_id];
    }

    // Guarantees that the global transform is available (unlike get_node).
    const float4x4& get_global_transform(SceneNode& node) {
        if (!node.global_transform) {
            if (node.parent != NODE_ID_INVALID) {
                assert(node_ids.is_used(node.parent));
                assert(scene_graph[node.parent]);
                node.global_transform =
                    mul(get_global_transform(*scene_graph[node.parent]), node.local_transform);
            } else {
                node.global_transform = node.local_transform;
            }
        }

        return node.global_transform.value();
    }

    // Guarantees that the global transform is available (unlike get_node).
    const float4x4& get_global_inverse_transposed_transform(SceneNode& node) {
        if (!node.global_inverse_transposed) {
            node.global_inverse_transposed = inverse(transpose(get_global_transform(node)));
        }

        return node.global_inverse_transposed.value();
    }

    // Guarantees that the global transform is available (unlike get_node).
    const float4x4& get_global_transform(const NodeID node_id) {
        assert(node_ids.is_used(node_id));
        assert(scene_graph[node_id]);
        return get_global_transform(*scene_graph[node_id]);
    }

    const float4x4& get_global_inverse_transposed_transform(const NodeID node_id) {
        assert(node_ids.is_used(node_id));
        assert(scene_graph[node_id]);
        return get_global_inverse_transposed_transform(*scene_graph[node_id]);
    }

    const Mesh& get_mesh(const MeshID mesh_id) {
        assert(mesh_ids.is_used(mesh_id));
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

    NodeID add_node(SceneNode node);

    void update_node(NodeID node_id, const float4x4& local_transform);

    void add_mesh_instance(MeshID mesh_id, NodeID node_id);

    void remove_mesh_instance(MeshID mesh_id, NodeID node_id);

    // Removes the mesh and its instances.
    void remove_mesh(MeshID mesh_id);

    // Removes the node, its children and attached instances.
    void remove_node(NodeID node_id);

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
    void rebuild_shader_object();

    // invalidates the global transform of this node and its children.
    void invalidate_node(SceneNode& node);

    // Groups meshes into BLASes according to the logic above.
    // All meshes in a group share the same instances (transforms); one group = one BLAS.
    void compute_mesh_groups();

    void ensure_pretransform_pipelines();

    // GPU pretransform: read src in model space, write dst in world space. The
    // dispatch records a memory barrier on the destination buffer.
    void pretransform_vertices_gpu(const CommandBufferHandle& cmd,
                                   const BufferHandle& src,
                                   const BufferHandle& dst,
                                   const float4x4& transform,
                                   const float4x4& inverse_transposed,
                                   uint32_t vertex_count);
    void pretransform_prev_vertices_gpu(const CommandBufferHandle& cmd,
                                        const BufferHandle& src,
                                        const BufferHandle& dst,
                                        const float4x4& transform,
                                        uint32_t vertex_count);

    // this only changes if the mesh groups changed or if a transform changes (TODO: selectively
    // upload transforms)
    void upload_geometry_data(const CommandBufferHandle& cmd);

    // this only changes if the mesh groups changed or if a transform changes (TODO: selectively
    // upload transforms)
    void upload_transforms(const CommandBufferHandle& cmd);

    // uploads the meshes, geometry data, and instance transforms
    void upload_meshes(const CommandBufferHandle& cmd);

    void build_blas(const CommandBufferHandle& cmd);
    void build_tlas(const CommandBufferHandle& cmd);

    void node_properties(Properties& props, const SceneNode& node);

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

    ASBuilder as_builder;

    // --------------------------
    // Scene definition

    MaterialSystemHandle material_system;
    FreeList<MeshID> mesh_ids;
    std::vector<MeshHandle> meshes; // sized to mesh_ids.size(); null at released slots
    FreeList<NodeID> node_ids;
    std::vector<std::optional<SceneNode>> scene_graph; // sized to node_ids.size()
    bool pretransform_dynamic = false;
    std::vector<CameraHandle> cameras;
    uint32_t active_camera = 0;
    AABB aabb; // can be invalid if information is not available.

    // Buffers dropped by remove_mesh; drained at the top of update() via
    // cmd->keep_until_pool_reset so in-flight cmds keep them alive.
    std::vector<BufferHandle> pending_buffer_releases;

    // --------------------------
    // Debug
    bool enable_debug_camera = false;
    CameraID debug_camera_id = CAMERA_ID_INVALID;

    // --------------------------
    // Cached and Precomputed

    bool needs_regroup = false;      // a mesh was instanced
    bool transforms_changed = false; // a node was updated
    bool tlas_dirty = false;

    inline static const MeshGroupID MESH_GROUP_ID_INVALID = MeshGroupID(-1);
    std::vector<MeshGroup> mesh_groups;
    // MeshID -> GroupID (MESH_GROUP_ID_INVALID if not in group, eg. because there was no instance
    // of the mesh)
    std::vector<MeshGroupID> mesh_to_group;

    // --------------------------
    // GPU data

    // Indexed with MeshID -> PrimitiveID
    std::vector<BufferHandle> index_buffers;
    // Indexed with MeshID -> indices
    std::vector<BufferHandle> vertex_buffers;
    // Indexed with MeshID -> indices. Null for static meshes.
    std::vector<BufferHandle> prev_vertex_buffers;

    std::vector<GeometryData> geometries;
    struct BLASGeometry {
        std::vector<vk::AccelerationStructureGeometryKHR> geometries;
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
    };
    std::vector<BLASGeometry> blas_geometries;

    std::vector<float4x4> instance_transforms;
    std::vector<float4x4> inverse_transposed_instance_transforms;
    std::vector<float4x4> prev_instance_transforms;
    std::vector<float4x4> prev_inverse_transposed_instance_transforms;

    // Indexed with GeometryID (InstanceID + GeometryIndex)
    BufferHandle geometries_buffer;
    BufferHandle instance_transforms_buffer;
    BufferHandle inverse_transposed_instance_transforms_buffer;
    BufferHandle prev_instance_transforms_buffer;
    BufferHandle prev_inverse_transposed_instance_transforms_buffer;

    std::vector<vk::AccelerationStructureInstanceKHR> tlas_instances;
    BufferHandle tlas_instances_buffer;
    BufferHandle as_scratch_buffer;
    AccelerationStructureHandle tlas;

    // GPU pretransform pipelines (lazily initialized).
    DescriptorSetLayoutHandle pretransform_descriptor_layout;
    PipelineHandle pretransform_vertex_pipeline;
    PipelineHandle pretransform_prev_vertex_pipeline;

    Camera prev_active_camera;
};

using SceneHandle = std::shared_ptr<Scene>;

} // namespace merian
