#pragma once

#include "merian-shaders/scene/env_map.hpp"
#include "merian-shaders/scene/light_collection.hpp"
#include "merian-shaders/scene/micromaps.hpp"
#include "merian-shaders/scene/scene-data.slangh"
#include "merian-shaders/shading/homogeneous_volume.hpp"
#include "merian-shaders/shading/materials/material_system.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/camera/camera.hpp"
#include "merian/utils/free_list.hpp"
#include "merian/utils/small_set.hpp"
#include "merian/vk/command/submission.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"
#include "merian/vk/memory/memory_suballocator_vma.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/raytrace/as_builder.hpp"

#include <initializer_list>
#include <optional>
#include <unordered_map>
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

    enum class MeshFlags : uint32_t {
        None = 0,
        // vertices can change between frames
        IsMorphed = 0x1,
        // Topology can change or vertices move incoherently between frames: rebuild, never refit.
        HasVariableTopology = 0x2,
        // default: treat all as non-opaque (allow alpha mask)
        IsOpaque = 0x4,
        // front winding is clockwise in object space
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

    // what a live GeometryID draws
    struct GeometryPlacement {
        MeshID mesh_id;
        NodeID node_id;

        bool operator==(const GeometryPlacement&) const = default;
    };

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

        // managed by Scene

        // Empty when invalidated; children must be invalidated too.
        std::optional<float4x4> global_transform;
        std::optional<float4x4> global_inverse_transposed;

        std::optional<float4x4> prev_global_transform;
        std::optional<float4x4> prev_global_inverse_transposed;

        bool transform_dirty = true;

        SmallSet<MeshID, 1> meshes;

        uint32_t serial = 0;
    };

    friend inline std::string format_as(const Node& node) {
        return fmt::format(
            "name: {}\nparent: {}\nnum children: {}\nlocal_transform:\n{}\nglobal_transform:\n{}",
            node.name.empty() ? "<none>" : node.name, node.parent, node.children.size(),
            node.local_transform, node.global_transform.value_or(float4x4(0)));
    }

    class Mesh {
      public:
        struct HostVertexSource {
            virtual ~HostVertexSource() = default;
            virtual void write(PackedVertexData* dst) const = 0;
        };

        using HostVertices = HostVertexSource*;

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

        // Non-owning: references a BufferHandle owned by the Mesh subclass.
        struct DeviceLocal {
            const BufferHandle& data;
        };

        using MeshVertexData =
            std::variant<HostVertices, HostPacked<PackedVertexData>, DeviceStaged, DeviceLocal>;

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

    // A suballocation inside one of Scene's shared vertex / index buffers.
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

    class MeshInfo {
      public:
        MeshHandle mesh;

        MeshBufferRegion vertex_buffer;
        // last frame's vertex_buffer while the vertices change, else empty
        MeshBufferRegion prev_vertex_buffer;
        MeshBufferRegion index_buffer;
        // of the uploaded buffers
        uint32_t primitive_count = 0;
        uint32_t uv_version = 0;

        SmallSet<NodeID, 1> instances;
        uint32_t animated_instance_count = 0;

        uint32_t serial = 0;

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

        // TLAS instance flags (GROUP_SPLIT_MASK); FlipFacing refers to the BLAS vertices
        MeshFlags flags;
        uint8_t instance_mask = 0xFF;
        bool has_animated_node = false;
        bool has_morphed_mesh = false;
        bool has_variable_topology_mesh = false;
        bool all_opaque = true;

        AccelerationStructureHandle blas;
        // the previous frame's BLAS while the group changes
        AccelerationStructureHandle spare_blas;
        vk::BuildAccelerationStructureFlagsKHR blas_build_flags{};
        // a mesh of the group changed
        bool blas_dirty = false;
        uint32_t blas_last_built_update = 0;
        uint32_t blas_last_updated_update = 0;
        std::optional<vk::AccelerationStructureBuildSizesInfoKHR> cached_blas_size_info;

        const SmallSet<NodeID, 1>& get_instances(const std::vector<MeshInfo>& mesh_infos) const {
            assert(!mesh_infos.empty());
            return mesh_infos[*this->meshes.begin()].instances;
        }

        // True if vertices are uploaded pretransformed and the TLAS instance transform is identity.
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
    // is_animated (transforms that change). A mesh is "dynamic" if either applies to any of
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
    //   dynamic (morphed OR animated)       -> not pretransformed, group by (split flags, NodeID)
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

    static DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info);

    static SlangCompositionHandle
    query_device_support_composition(const DeviceSupportQueryInfo& query_info);

    // frame 0 starts a run: restarts the learned light state and bakes all pending micromaps
    void update(Submission& submission,
                float time,
                float time_diff,
                uint32_t frame,
                uint32_t seed,
                const ShaderObjectAllocatorHandle& obj_allocator = nullptr);

    enum SceneUpdateTypeBits : uint32_t {
        // a mesh or instance was added or removed
        SceneUpdateGeometry = 1u << 0,
        SceneUpdateTransform = 1u << 1,
        // the active camera moved or a different one became active
        SceneUpdateCamera = 1u << 2,
    };

    uint32_t get_last_updates() const {
        return last_updates;
    }

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    const MaterialSystemHandle& get_material_system() const {
        return material_system;
    }

    const EnvMapHandle& get_env() const {
        return env_map;
    }

    LightCollection& get_light_collection() {
        return lights;
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

    uint32_t get_active_camera_index() const {
        return active_camera;
    }

    // Must be true before binding the shader object.
    virtual bool is_ready() const {
        return false;
    }

    // nullopt if there is no active camera or it has no resolution opinion
    std::optional<vk::Extent3D> get_resolution() const;

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

    // indexed with this frame's live GeometryIDs
    const std::vector<GeometryPlacement>& get_geometry_placements() const {
        return geometry.placements;
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

    // The medium filling the scene outside of any surface. Swapping the implementation swaps the
    // type the renderers compile against, so Vacuum removes the medium rather than branching it.
    void set_exterior_volume(const HomogeneousVolumeHandle& volume);

    const HomogeneousVolumeHandle& get_exterior_volume() const {
        return exterior_volume;
    }

    // False while the scene is vacuum, i.e. while the medium is compiled out of the renderers.
    bool has_exterior_volume() const {
        return exterior_volume && exterior_volume->get_type_name() != "merian::Vacuum";
    }

    // A the scenes up direction
    virtual float3 get_up() {
        return float3(0, 0, 1);
    }

    // Scene-time hook. The default returns the update time unchanged. Scenes
    // that own a separate clock can override.
    virtual float get_time(float time) {
        return time;
    }

    void properties(Properties& props);

    const AABB& get_aabb() const {
        return aabb;
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

  protected:
    virtual void
    on_update(const CommandBufferHandle& cmd, float time, float time_diff, uint32_t frame) {
        (void)cmd;
        (void)time;
        (void)time_diff;
        (void)frame;
    }

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

    // Releases the object once the frames that may still read it have completed.
    void retire(ConstObjectHandle object);

  private:
    ShaderObjectHandle build_shader_object() const;
    void drain_retired(const CommandBufferHandle& cmd);

    void invalidate_node(Node& node);

    // one group = one BLAS; all meshes of a group share the same instances
    void compute_mesh_groups();

    void ensure_transform_pipelines();

    void stage_transform_jobs(const CommandBufferHandle& cmd);

    void dispatch_batched_transforms(const CommandBufferHandle& cmd);

    void upload_geometry_data(const CommandBufferHandle& cmd);

    // vanished geometries map to appended Retired entries, unseen ones are marked New
    void compute_prev_geometry_remap();

    void upload_prev_geometry_remap(const CommandBufferHandle& cmd);

    void upload_transforms(const CommandBufferHandle& cmd);

    void upload_meshes(const CommandBufferHandle& cmd);

    void build_blas(const CommandBufferHandle& cmd);

    void ensure_micromaps(Submission& submission,
                          const ShaderObjectAllocatorHandle& obj_allocator,
                          const bool complete);
    void build_tlas(const CommandBufferHandle& cmd);

    // re-suballocates every live region into a new backing buffer and copies it over
    void reallocate_shared_buffer(const CommandBufferHandle& cmd,
                                  VMAMemorySubAllocatorHandle& slot,
                                  vk::DeviceSize& capacity_slot,
                                  std::initializer_list<MeshBufferRegion MeshInfo::*> region_fields,
                                  vk::DeviceSize new_capacity,
                                  vk::BufferUsageFlags usage,
                                  const std::string& debug_name);

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
    void properties_volume(Properties& props);
    void properties_lights(Properties& props);

    void process_pending_env_load(const CommandBufferHandle& cmd);

    // selects the ScenePrevEnvMap type of prev_env_map
    void compose_prev_env();

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

    // Selects the SceneHomogeneousVolume type; swapping it recompiles the shaders.
    HomogeneousVolumeHandle exterior_volume;

    ASBuilder as_builder;
    bool as_supported = false;

    // --- Scene definition ---

    MaterialSystemHandle material_system;
    EnvMapHandle env_map;
    // as written by the last update
    EnvMapHandle prev_env_map;
    std::string prev_env_type_name;
    LightCollection lights;
    std::vector<LightCollection::EmissiveGeometry> emissive_geometries;

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
    uint32_t update_count = 0;
    bool starting_run = true;

    std::unique_ptr<Micromaps> micromaps;
    ShaderObjectAllocatorHandle fallback_obj_allocator;
    bool opacity_micromaps_enabled = true;
    bool emission_micromaps_enabled = true;

    uint32_t last_updates = 0;

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
        vk::DeviceSize gpu_transform_buffer_bytes = 0;

        uint32_t blas_builds = 0;
        uint32_t blas_builds_static = 0;
        uint32_t blas_builds_dynamic = 0;
        uint32_t blas_updates = 0;

        bool tlas_rebuilt = false;
        uint32_t tlas_instance_count = 0;

        uint32_t buffers_allocated = 0;
        uint32_t objects_retired = 0;

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
    bool store_cameras = false;
    AABB aabb; // can be invalid if information is not available.

    std::vector<ConstObjectHandle> retired;

    bool needs_regroup = false;      // a mesh was instanced
    bool transforms_changed = false; // a node was updated
    bool tlas_dirty = false;

    inline static const MeshGroupID MESH_GROUP_ID_INVALID = MeshGroupID(-1);
    std::vector<MeshGroup> mesh_groups;
    // MeshID -> GroupID; MESH_GROUP_ID_INVALID for meshes with no instances.
    std::vector<MeshGroupID> mesh_to_group;

    // Shared scene buffers: one backing buffer per kind, suballocated per-mesh.
    VMAMemorySubAllocatorHandle shared_vb_suballoc;
    VMAMemorySubAllocatorHandle shared_ib_suballoc;
    vk::DeviceSize shared_vb_capacity = 0;
    vk::DeviceSize shared_ib_capacity = 0;

    // live geometries first, then the Retired ones only previous-frame hits reach
    struct FrameGeometry {
        AccelerationStructureHandle tlas;
        std::vector<GeometryData> geometries;
        // of the live geometries
        std::vector<uint64_t> geometry_keys;
        // GeometryID -> instance index
        std::vector<uint32_t> instance_indices;
        // of the live geometries
        std::vector<GeometryPlacement> placements;
        std::vector<float4x4> instance_transforms;
        std::vector<float4x4> inverse_transposed_instance_transforms;
        std::vector<float4x4> prev_instance_transforms;
        std::vector<float4x4> prev_inverse_transposed_instance_transforms;
    };
    FrameGeometry geometry;
    FrameGeometry prev_geometry;

    std::vector<GeometryRemap> prev_geometry_remap;
    std::unordered_map<uint64_t, GeometryID> geometry_of_key;
    uint32_t next_serial = 0;
    struct BLASGeometry {
        std::vector<vk::AccelerationStructureGeometryKHR> geometries;
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
        std::vector<vk::AccelerationStructureTrianglesOpacityMicromapEXT> micromaps;
    };
    std::vector<BLASGeometry> blas_geometries;

    // Indexed with GeometryID (InstanceID + GeometryIndex)
    BufferHandle geometries_buffer;
    BufferHandle instance_transforms_buffer;
    BufferHandle inverse_transposed_instance_transforms_buffer;
    BufferHandle prev_instance_transforms_buffer;
    BufferHandle prev_inverse_transposed_instance_transforms_buffer;
    BufferHandle prev_geometry_remap_buffer;

    std::vector<vk::AccelerationStructureInstanceKHR> tlas_instances;
    BufferHandle tlas_instances_buffer;
    BufferHandle as_scratch_buffer;

    DescriptorSetLayoutHandle transform_descriptor_layout;
    PipelineHandle transform_vertex_pipeline;
    BufferHandle transform_vertex_job_buf;
    std::vector<TransformVertexJob> vertex_jobs;
    uint32_t max_vertex_count_in_jobs = 0;

    Camera prev_active_camera;
    std::optional<float> prev_time;
};

using SceneHandle = std::shared_ptr<Scene>;

// area-weighted, (0, 1, 0) where the adjacent triangles have no area
std::vector<float3> compute_vertex_normals(const std::vector<float3>& positions,
                                           const std::vector<uint3>& triangles);

} // namespace merian
