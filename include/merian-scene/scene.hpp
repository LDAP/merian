#pragma once

#include "merian-scene/env_map.hpp"
#include "merian-shaders/scene/scene-data.slangh"
#include "merian-scene/material_system.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/camera/camera.hpp"
#include "merian/utils/free_list.hpp"
#include "merian/utils/small_set.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/memory/frame_staging_block.hpp"
#include "merian/vk/memory/memory_suballocator_vma.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/raytrace/as_builder.hpp"

#include <optional>
#include <variant>
#include <vector>

namespace merian {

class Scene : public std::enable_shared_from_this<Scene> {
  public:
    // --- IDs ---

    using NodeID = uint32_t;
    using MeshID = uint32_t;
    using CameraID = uint32_t;

    // Root nodes use NODE_ID_INVALID as parent; their local_transform is the global transform.
    static constexpr NodeID NODE_ID_INVALID = UINT32_MAX;
    static constexpr CameraID CAMERA_ID_INVALID = UINT32_MAX;

    // --- Mesh flags ---

    enum class MeshFlags : uint32_t {
        None = 0,
        // default: not; means: vertices can move over time, get_prev_vertices must be valid.
        IsMorphed = 0x1,
        // Topology (vertex count, primitive count) can change between frames.
        HasVariableTopology = 0x2,
        // default: treat all as non-opaque (allow alpha mask)
        IsOpaque = 0x4,
        // Set when the mesh's front winding is clockwise (default is counter-clockwise).
        // Maps to VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR.
        FlipFacing = 0x8,
        // default: cull backfaces
        TwoSided = 0x10,
        // Per-vertex tangents are meaningful; otherwise consumers derive from UVs.
        HasTangents = 0x20,
        // Use the geometric face normal as the shading normal (hard edges, low-poly look).
        FlatShading = 0x40,
        // Replace surface shading with the scene env map sampled along the ray.
        UseEnvMap = 0x80,
    };

    friend constexpr MeshFlags operator|(MeshFlags a, MeshFlags b) {
        return static_cast<MeshFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    friend constexpr bool operator&(MeshFlags a, MeshFlags b) {
        return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
    }

    friend inline std::string format_as(const MeshFlags flags) {
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
        if (flags & MeshFlags::FlipFacing) {
            append("FlipFacing");
        }
        if (flags & MeshFlags::TwoSided) {
            append("TwoSided");
        }
        if (flags & MeshFlags::HasVariableTopology) {
            append("HasVariableTopology");
        }
        if (flags & MeshFlags::HasTangents) {
            append("HasTangents");
        }
        if (flags & MeshFlags::FlatShading) {
            append("FlatShading");
        }
        if (flags & MeshFlags::UseEnvMap) {
            append("UseEnvMap");
        }
        return out;
    }

    // --- Instance ---

    // A mesh placed at a scene node.
    struct Instance {
        NodeID node_id;
        uint8_t mask;

        Instance(NodeID node_id, uint8_t mask = 0x01) : node_id(node_id), mask(mask) {}

        auto operator<=>(const Instance&) const = default;
    };

    // --- Node ---

    class Node {
      public:
        std::string name;

        bool is_animated = false; // transform can change between frames

        NodeID parent = NODE_ID_INVALID;
        std::vector<NodeID> children;

        float4x4 local_transform = identity();

        // --- Managed by Scene ---

        // Empty when invalidated; children must be invalidated too.
        std::optional<float4x4> global_transform;
        std::optional<float4x4> global_inverse_transposed;

        std::optional<float4x4> prev_global_transform;
        std::optional<float4x4> prev_global_inverse_transposed;

        bool transform_dirty = true;
    };

    friend inline std::string format_as(const Node& node) {
        return fmt::format(
            "name: {}\nparent: {}\nnum children: {}\nlocal_transform:\n{}\nglobal_transform:\n{}",
            node.name.empty() ? "<none>" : node.name, node.parent, node.children.size(),
            node.local_transform, node.global_transform.value_or(float4x4(0)));
    }

    // --- Mesh ---

    class Mesh {
      public:
        // ----- Bulk CPU vertex data -----
        template <typename T> struct HostVertexSource {
            virtual ~HostVertexSource() = default;
            virtual void write(T* dst) const = 0;
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
        // Non-owning: references a BufferHandle owned by the Mesh subclass.
        struct DeviceStaged {
            const BufferHandle& data;
            vk::DeviceSize offset = 0;
        };

        // ----- GPU Local Data (Procedural/Static GPU data) -----
        // Non-owning: references a BufferHandle owned by the Mesh subclass.
        struct DeviceLocal {
            const BufferHandle& data;
        };

        using MeshVertexData =
            std::variant<HostVertices, HostPacked<PackedVertexData>, DeviceStaged, DeviceLocal>;

        // can be std::monostate for static meshes.
        using MeshPrevVertexData = std::variant<std::monostate,
                                                HostPrevVertices,
                                                HostPacked<PackedPrevVertexData>,
                                                DeviceStaged,
                                                DeviceLocal>;

        // monostate means no index buffer (only valid when index_type == eNoneKHR).
        using MeshIndexData =
            std::variant<std::monostate, HostIndices, HostPacked<void>, DeviceStaged, DeviceLocal>;

      public:
        std::string name;
        MaterialID material_id{};
        MeshFlags flags = MeshFlags::IsOpaque;
        uint8_t instance_mask = 0x01;
        vk::IndexType index_type = vk::IndexType::eUint32;

        // Dirty triggers reupload/copy + TLAS rebuild; cost depends on the data variant.
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

        bool has_variable_topology() const {
            return flags & MeshFlags::HasVariableTopology;
        }

        bool is_flip_facing() const {
            return flags & MeshFlags::FlipFacing;
        }

        bool is_two_sided() const {
            // if yes, needs to disable backface culling
            return flags & MeshFlags::TwoSided;
        }

        bool is_opaque() const {
            // if yes, allows to set the force opaque flag when raytracing.
            return flags & MeshFlags::IsOpaque;
        }

        // false: primitive i has vertices (3i, 3i+1, 3i+2); no index buffer is allocated.
        bool has_indices() const {
            return index_type != vk::IndexType::eNoneKHR;
        }

        bool is_dirty() const {
            return vertices_dirty || indices_dirty;
        }
    };

    using MeshHandle = std::unique_ptr<Mesh>;

    friend inline std::string format_as(const Mesh& mesh) {
        return fmt::format("vertices: {}\ntriangles: {}\nmaterial id: {}\nflags: {}\n"
                           "instance mask: 0x{:02x}\nindex type: {}\n"
                           "vertices dirty: {}\nindices dirty: {}",
                           mesh.get_vertex_count(), mesh.get_primitive_count(), mesh.material_id,
                           mesh.flags, mesh.instance_mask, vk::to_string(mesh.index_type),
                           mesh.vertices_dirty, mesh.indices_dirty);
    }

    // A suballocation inside one of Scene's shared vertex / prev-vertex / index buffers.
    struct MeshBufferRegion {
        MemoryAllocationHandle suballoc;
        vk::DeviceSize size = 0;
        vk::DeviceSize alignment = 1;

        vk::DeviceAddress get_device_address() const {
            if (!suballoc)
                return 0;
            const auto* sub = static_cast<const VMAMemorySubAllocation*>(suballoc.get());
            return sub->get_suballocator()->get_base_buffer()->get_device_address() +
                   sub->get_offset();
        }

        vk::DeviceSize get_offset() const {
            if (!suballoc)
                return 0;
            return static_cast<const VMAMemorySubAllocation*>(suballoc.get())->get_offset();
        }

        explicit operator bool() const {
            return static_cast<bool>(suballoc) && size > 0;
        }
    };

    // Scene-owned per-mesh record.
    class MeshInfo {
      public:
        MeshHandle mesh;

        MeshBufferRegion vertex_buffer;
        MeshBufferRegion prev_vertex_buffer;
        MeshBufferRegion index_buffer;

        SmallSet<NodeID, 1> instances;
        uint32_t animated_instance_count = 0;

        bool is_dynamic() const {
            return mesh && (mesh->is_morphed() || animated_instance_count > 0);
        }

        bool is_static() const {
            return !is_dynamic();
        }
    };

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

        MeshVertexData get_vertices() const override {
            return HostPacked<PackedVertexData>{vertices.data()};
        }
        MeshPrevVertexData get_prev_vertices() const override {
            if (!is_morphed())
                return std::monostate{};
            assert(prev_vertices.size() == vertices.size());
            return HostPacked<PackedPrevVertexData>{prev_vertices.data()};
        }
        MeshIndexData get_indices() const override {
            assert(index_type == vk::IndexType::eUint32);
            return HostPacked<void>{indices.data()};
        }
    };

    class Error : public std::runtime_error {
      public:
        Error(const std::string& msg) : std::runtime_error(msg) {}
    };

  private:
    // --- Internal types ---

    using MeshGroupID = uint32_t;

    // Flags that map to TLAS instance flags and therefore force group splits.
    static constexpr MeshFlags GROUP_SPLIT_MASK = static_cast<MeshFlags>(
        static_cast<uint32_t>(MeshFlags::FlipFacing) | static_cast<uint32_t>(MeshFlags::TwoSided));
    static_assert(static_cast<uint32_t>(GROUP_SPLIT_MASK) < (1u << 16),
                  "split_key reserves bits 16-23 for instance_mask");

    struct MeshGroup {
        SmallSet<MeshID, 1> meshes;

        // only the TLAS-instance-level flags shared by all meshes (GROUP_SPLIT_MASK)
        MeshFlags flags;
        uint8_t instance_mask = 0xFF;
        // true if any instance node has is_animated set
        bool has_animated_node = false;
        // true if any mesh in the group is morphed
        bool has_morphed_mesh = false;
        // true if any mesh in the group has HasVariableTopology
        bool has_variable_topology_mesh = false;
        // true if all meshes in the group are opaque
        bool all_opaque = true;

        // ----------------
        AccelerationStructureHandle blas;
        vk::BuildAccelerationStructureFlagsKHR blas_build_flags{};
        // a mesh changed (new mesh in group -> not the same group, never true)
        bool blas_dirty = false;
        uint32_t blas_last_built_frame = 0;
        uint32_t blas_last_updated_frame = 0;
        std::optional<vk::AccelerationStructureBuildSizesInfoKHR> cached_blas_size_info;
        // ----------------

        const SmallSet<NodeID, 1>& get_instances(const std::vector<MeshInfo>& mesh_infos) const {
            assert(!mesh_infos.empty());
            return mesh_infos[*this->meshes.begin()].instances;
        }

        // True if vertices are uploaded pretransformed and the TLAS instance transform is identity.
        // TODO: pretransforming animated non-morphed meshes makes them effectively morphed and
        //       requires a prev vertex buffer for motion vectors.
        bool is_pretransformed(const std::vector<MeshInfo>& mesh_infos,
                               bool pretransform_animated) const {
            assert(!mesh_infos.empty());
            const MeshInfo& info = mesh_infos[*this->meshes.begin()];
            if (info.instances.size() > 1)
                return false;
            if ((has_animated_node || has_morphed_mesh) && !pretransform_animated)
                return false;
            return true;
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
    // BLASs (one per group, split by GROUP_SPLIT_MASK = FrontCCW | TwoSided)
    // IsOpaque is per-geometry; IsMorphed is per-mesh — neither splits groups.
    //
    // pretransform_animated ON:
    //   static (non-morphed, non-animated)  -> pretransformed, group by split flags, built once
    //   dynamic (morphed OR animated)       -> pretransformed, group by split flags, rebuilt
    //   instanced                           -> group by (instance set, split flags)
    // pretransform_animated OFF:
    //   static non-morphed non-animated     -> pretransformed, group by split flags, built once
    //   morphed non-animated                -> not pretransformed, group by split flags
    //   animated non-instanced              -> not pretransformed, group by (split flags, NodeID)
    //   instanced                           -> group by (instance set, split flags)
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
          const MaterialSystemHandle& material_system);

    virtual ~Scene() = default;

    // ------------------------------

    void update(const CommandBufferHandle& cmd, float time, float time_diff, uint32_t frame);

    // What changed during the most recent update(). Lets consumers (e.g. temporal
    // accumulation) react to scene edits without diffing the scene themselves.
    struct UpdateChanges {
        bool geometry_changed = false;  // a mesh or instance was added or removed
        bool transform_changed = false; // a node transform changed
        bool camera_changed = false;    // the active camera moved or a different one became active
    };

    const UpdateChanges& get_last_update_changes() const {
        return last_update_changes;
    }

    // ------------------------------

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    const MaterialSystemHandle& get_material_system() const {
        return material_system;
    }

    const EnvMapHandle& get_env() const {
        return env_map;
    }

    void set_env(EnvMapHandle env);

    const TextureManagerHandle& get_texture_manager() const {
        return material_system->get_texture_manager();
    }

    // Caller must check is_ready() first — otherwise resources may be stale or unset.
    const ShaderObjectHandle& get_shader_object() const {
        return shader_object.get();
    }

    operator const ShaderObjectHandle&() const {
        return shader_object.get();
    }

    std::vector<CameraHandle> get_cameras() const;

    CameraHandle get_camera(const CameraID camera_id) const;

    CameraHandle get_active_camera() const;

    // Must be true before binding the shader object. Subclasses override.
    virtual bool is_ready() const {
        return false;
    }

    void set_active_camera(uint32_t index);

    bool has_geometry() const {
        return !mesh_infos.empty();
    }

    const std::vector<std::optional<Node>>& get_scene_graph() const {
        return scene_graph;
    }

    const std::vector<MeshInfo>& get_mesh_infos() const {
        return mesh_infos;
    }

    // to get transforms use get_*_transform(...)
    const Node& get_node(const NodeID node_id) {
        assert(node_ids.is_used(node_id));
        return *scene_graph[node_id];
    }

    // Guarantees that the global transform is available (unlike get_node).
    const float4x4& get_global_transform(Node& node) {
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
    const float4x4& get_global_inverse_transposed_transform(Node& node) {
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

    const MeshInfo& get_mesh(const MeshID mesh_id) {
        assert(mesh_ids.is_used(mesh_id));
        return mesh_infos[mesh_id];
    }

    // Bake single-instance animated (node transforms can change) meshes world transforms on CPU at
    // upload time. Static (non-animated) meshes are always pre-transformed.
    bool get_pretransform_animated() const {
        return pretransform_animated;
    }
    void set_pretransform_animated(bool value);

    // Removes meshes, nodes, cameras and resets the AABB. Not: env map, materials and textures.
    void clear_geometry();

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

    NodeID add_node(Node node);

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

    void defer_buffer_release(BufferHandle buffer) {
        if (buffer)
            pending_buffer_releases.push_back(std::move(buffer));
    }

  private:
    ShaderObjectHandle build_shader_object() const;

    // invalidates the global transform of this node and its children and marks the node as dirty.
    void invalidate_node(Node& node);

    // Groups meshes into BLASes according to the logic above.
    // All meshes in a group share the same instances (transforms); one group = one BLAS.
    void compute_mesh_groups();

    // Whether this mesh in this group needs a prev_vertex_buffer for motion vectors.
    bool needs_prev_vertices(const MeshGroup& group, const MeshInfo& info) const;

    void ensure_transform_pipelines();

    // Upload accumulated TransformVertexJob / TransformPrevVertexJob arrays to their
    // device buffers. Called before the transfer→compute barrier.
    void stage_transform_jobs(const CommandBufferHandle& cmd);

    // Bind + push descriptor + dispatch the two batched transform shaders. No transfers,
    // no internal barrier; the two dispatches touch disjoint per-mesh buffers.
    void dispatch_batched_transforms(const CommandBufferHandle& cmd);

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

    // Grows one shared buffer: re-suballocates every live region into a new backing buffer and
    // copies the existing data over in a single vkCmdCopyBuffer.
    void reallocate_shared_buffer(const CommandBufferHandle& cmd,
                                  VMAMemorySubAllocatorHandle& slot,
                                  vk::DeviceSize& capacity_slot,
                                  MeshBufferRegion MeshInfo::* region_field,
                                  vk::DeviceSize new_capacity,
                                  vk::BufferUsageFlags usage,
                                  const std::string& debug_name);

    // Detail panes that double as drill-down targets from explorer lists.
    void properties_node(Properties& props, NodeID node_id);
    void properties_mesh(Properties& props, MeshID mesh_id);
    void properties_mesh_group(Properties& props, MeshGroupID group_id);

    // Explorer sub-panes.
    void properties_explorer(Properties& props);
    void properties_cameras(Properties& props);
    void properties_meshes(Properties& props);
    void properties_mesh_groups(Properties& props);
    void properties_graph(Properties& props);

    // Top-level sections.
    void properties_settings(Properties& props);
    void properties_statistics(Properties& props);
    void properties_env(Properties& props);

    void process_pending_env_load(const CommandBufferHandle& cmd);

  private:
    // --- Context ---

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;

    SlangCompositionHandle composition;
    Versioned<SlangProgram> layout_program;
    Versioned<ShaderObject> shader_object;

    // Tracks the merian_hint_enable_thin_lens composition constant; toggling recompiles shaders.
    bool thin_lens_enabled = false;
    void set_enable_thin_lens(const bool enable);

    ASBuilder as_builder;
    bool as_supported = false;

    // --- Scene definition ---

    MaterialSystemHandle material_system;
    EnvMapHandle env_map;

    struct EnvUIState {
        int selection = 0;
        std::string latlong_path;
        std::array<std::string, 6> cubemap_paths;
    };
    EnvUIState env_ui;

    struct PendingEnvLoad {
        int kind;
        std::string latlong_path;
        std::array<std::string, 6> cubemap_paths;
    };
    std::optional<PendingEnvLoad> pending_env_load;
    FreeList<MeshID> mesh_ids;
    // Sized to mesh_ids.size(). Released slots have a null mesh.
    std::vector<MeshInfo> mesh_infos;
    FreeList<NodeID> node_ids;
    std::vector<std::optional<Node>> scene_graph; // sized to node_ids.size()
    bool pretransform_animated = false;
    float blas_rebuild_fraction = 0.33f;
    uint32_t current_frame = 0;

    UpdateChanges last_update_changes;

    struct FrameStats {
        uint32_t meshes_uploaded_device_local = 0;
        uint32_t meshes_uploaded_device_staged = 0;
        uint32_t meshes_uploaded_host_packed = 0;
        uint32_t meshes_uploaded_host_unpacked = 0;
        uint32_t vertices_uploaded = 0;
        uint32_t indices_uploaded = 0;
        vk::DeviceSize upload_bytes = 0;

        uint32_t gpu_vertex_transforms = 0;
        uint32_t gpu_vertex_transform_vertices = 0;
        uint32_t gpu_prev_vertex_transforms = 0;
        uint32_t gpu_prev_vertex_transform_vertices = 0;
        vk::DeviceSize gpu_transform_buffer_bytes = 0;

        uint32_t blas_builds = 0;
        uint32_t blas_builds_static = 0;
        uint32_t blas_builds_dynamic = 0;
        uint32_t blas_updates = 0;

        bool tlas_rebuilt = false;
        uint32_t tlas_instance_count = 0;

        uint32_t buffers_allocated = 0;
        uint32_t buffers_released = 0;

        vk::DeviceSize geometry_data_bytes = 0;
        vk::DeviceSize transform_data_bytes = 0;
        vk::DeviceSize tlas_instance_data_bytes = 0;

        uint32_t meshes_uploaded() const {
            return meshes_uploaded_device_local + meshes_uploaded_device_staged +
                   meshes_uploaded_host_packed + meshes_uploaded_host_unpacked;
        }
    };

    FrameStats frame_stats{};

    std::vector<CameraHandle> cameras;
    uint32_t active_camera = 0;
    AABB aabb; // can be invalid if information is not available.

    // Buffers dropped by remove_mesh; drained at the top of update() via
    // cmd->keep_until_pool_reset so in-flight cmds keep them alive.
    std::vector<BufferHandle> pending_buffer_releases;
    std::vector<AccelerationStructureHandle> pending_blas_releases;

    // --- Cached and precomputed ---

    bool needs_regroup = false;      // a mesh was instanced
    bool transforms_changed = false; // a node was updated
    bool transforms_changed_last_frame = false;
    bool tlas_dirty = false;

    inline static const MeshGroupID MESH_GROUP_ID_INVALID = MeshGroupID(-1);
    std::vector<MeshGroup> mesh_groups;
    // MeshID -> GroupID; MESH_GROUP_ID_INVALID for meshes with no instances.
    std::vector<MeshGroupID> mesh_to_group;

    // --- GPU data ---

    // Shared scene buffers: one backing buffer per kind, suballocated per-mesh.
    VMAMemorySubAllocatorHandle shared_vb_suballoc;
    VMAMemorySubAllocatorHandle shared_prev_vb_suballoc;
    VMAMemorySubAllocatorHandle shared_ib_suballoc;
    vk::DeviceSize shared_vb_capacity = 0;
    vk::DeviceSize shared_prev_vb_capacity = 0;
    vk::DeviceSize shared_ib_capacity = 0;

    std::optional<FrameStagingBlock> frame_staging;

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

    // Batched GPU transform pipelines (lazily initialized) and reusable device buffers.
    DescriptorSetLayoutHandle transform_descriptor_layout;
    PipelineHandle transform_vertex_pipeline;
    PipelineHandle transform_prev_vertex_pipeline;
    BufferHandle transform_vertex_job_buf;
    BufferHandle transform_prev_vertex_job_buf;
    std::vector<TransformVertexJob> vertex_jobs;
    std::vector<TransformPrevVertexJob> prev_vertex_jobs;
    uint32_t max_vertex_count_in_jobs = 0;
    uint32_t max_prev_vertex_count_in_jobs = 0;

    Camera prev_active_camera;
};

using SceneHandle = std::shared_ptr<Scene>;

} // namespace merian
