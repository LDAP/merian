#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/slang_program.hpp"
#include "merian/utils/normal_encoding.hpp"

#include <algorithm>
#include <cassert>
#include <fmt/format.h>
#include <map>
#include <unordered_map>

namespace merian {

namespace {

uint32_t pack_tangent(const float3 tangent_dir, const float sign) {
    const uint32_t enc = encode_normal(tangent_dir);
    return (enc & ~1u) | (sign < 0.f ? 1u : 0u);
}

} // namespace

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

PackedVertexData Mesh::get_packed_vertex(uint32_t vertex_idx) const {
    PackedVertexData v;
    v.position = get_position(vertex_idx);
    v.encoded_normal = encode_normal(get_normal(vertex_idx));
    const float2 uv = get_uv(vertex_idx);
    v.uv = half2(uv.x, uv.y);
    const float4 t = get_tangent(vertex_idx);
    v.encoded_tangent = pack_tangent(float3(t.x, t.y, t.z), t.w);
    return v;
}

PackedVertexData Mesh::get_packed_vertex_pretransformed(uint32_t vertex_idx,
                                                        const SceneNode& node) const {
    assert(node.global_transform && node.global_inverse_transposed);
    const float4x4& m = *node.global_transform;
    const float4x4& it = *node.global_inverse_transposed;

    PackedVertexData v;
    v.position = mul(m, float4(get_position(vertex_idx), 1.f));
    v.encoded_normal =
        encode_normal(normalize(float3(mul(it, float4(get_normal(vertex_idx), 0.f)))));
    const float2 uv = get_uv(vertex_idx);
    v.uv = half2(uv.x, uv.y);
    const float4 t = get_tangent(vertex_idx);
    const float3 tw = normalize(float3(mul(it, float4(t.x, t.y, t.z, 0.f))));
    v.encoded_tangent = pack_tangent(tw, t.w);
    return v;
}

PackedPrevVertexData Mesh::get_packed_prev_vertex(uint32_t vertex_idx) const {
    return PackedPrevVertexData{.position = get_prev_position(vertex_idx)};
}

PackedPrevVertexData Mesh::get_packed_prev_vertex_pretransformed(uint32_t vertex_idx,
                                                                 const SceneNode& node) const {
    assert(node.global_transform);
    const float4x4& m = *node.global_transform;

    return PackedPrevVertexData{.position = mul(m, float4(get_prev_position(vertex_idx), 1))};
}

float3 SimpleMesh::get_normal(uint32_t vertex_idx) const {
    return decode_normal(vertices[vertex_idx].encoded_normal);
}

float4 SimpleMesh::get_tangent(uint32_t vertex_idx) const {
    const uint32_t enc = vertices[vertex_idx].encoded_tangent;
    const float3 t = decode_normal(enc & ~1u);
    const float sign = (enc & 1u) ? -1.f : 1.f;
    return float4(t.x, t.y, t.z, sign);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Scene::Scene(const ShaderCompileContextHandle& compile_context,
             const ContextHandle& context,
             const ResourceAllocatorHandle& allocator,
             const ShaderObjectAllocatorHandle& obj_allocator,
             const MaterialSystemHandle& material_system)
    : compile_context(compile_context), context(context), allocator(allocator),
      obj_allocator(obj_allocator), as_builder(context, allocator),
      material_system(material_system) {

    assert(context->get_device()
               ->get_enabled_features()
               .get_acceleration_structure_features_khr()
               .accelerationStructure == VK_TRUE);

    // Build composition once — subsequent changes modify in-place.
    composition = SlangComposition::create();
    composition->add_composition(material_system->get_composition());
    composition->add_module_from_path("merian-shaders/scene/scene.slang");
    composition->add_module_from_path("merian-shaders/scene/camera.slang");
    composition->add_module_from_path("merian-shaders/scene/environment-map.slang");
    composition->add_module_from_path("merian-shaders/scene/acceleration-structure.slang");

    rebuild_shader_object();

    // TODO: use link-time type for AS once Slang's lookupExternDeclRefType is fixed
    // if (build_as) {
    //     composition->add_module_from_string(
    //         "scene_as_type",
    //         "import merian_shaders.scene.acceleration_structure;\n"
    //         "namespace merian { export struct SceneAccelerationStructure : AccelerationStructure
    //         "
    //         "= HWAccelerationStructure; }");
    // } else {
    //     composition->add_module_from_string(
    //         "scene_as_type",
    //         "import merian_shaders.scene.acceleration_structure;\n"
    //         "namespace merian { export struct SceneAccelerationStructure : AccelerationStructure
    //         "
    //         "= NullAccelerationStructure; }");
    // }
}

void Scene::rebuild_shader_object() {
    if (!layout_program) {
        layout_program = SlangProgram::create(compile_context, composition);
    }

    shader_object = layout_program->create_shader_object(context, "merian::Scene", obj_allocator);

    auto c = shader_object->get_cursor();

    c["material_system"] = material_system;
    material_system->on_changed(shader_object, [&] {
        auto c = shader_object->get_cursor();
        c["material_system"] = material_system;
    });

    c["geometries"] = geometries_buffer ? geometries_buffer : allocator->get_dummy_buffer();

    c["instance_transforms"] =
        instance_transforms_buffer ? instance_transforms_buffer : allocator->get_dummy_buffer();
    c["inverse_transposed_instance_transforms"] =
        inverse_transposed_instance_transforms_buffer
            ? inverse_transposed_instance_transforms_buffer
            : allocator->get_dummy_buffer();
    c["prev_instance_transforms"] = prev_instance_transforms_buffer
                                        ? prev_instance_transforms_buffer
                                        : allocator->get_dummy_buffer();
    c["prev_inverse_transposed_instance_transforms"] =
        prev_inverse_transposed_instance_transforms_buffer
            ? prev_inverse_transposed_instance_transforms_buffer
            : allocator->get_dummy_buffer();
}

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------

// bool Scene::set_build_acceleration_structure(bool build) {
//     if (build_as == build)
//         return false;
//     build_as = build;
//     // TODO: rebuild composition once link-time AS type is re-enabled
//     return true;
// }

MeshID Scene::add_mesh(MeshHandle mesh) {
    assert(mesh);
    const MeshID id = mesh_ids.acquire();
    if (id >= meshes.size()) {
        meshes.resize(mesh_ids.size());
    }
    meshes[id] = std::move(mesh);

    // note we ignore the mesh as long there are no instances
    // in add_mesh_instance the mesh is marked dirty and regrouping is enfored.

    return id;
}

NodeID Scene::add_node(SceneNode node) {
    const NodeID id = node_ids.acquire();
    if (id >= scene_graph.size()) {
        scene_graph.resize(node_ids.size());
    }
    if (node.parent != NODE_ID_INVALID) {
        assert(node_ids.is_used(node.parent));

        SceneNode& parent = *scene_graph[node.parent];
        parent.children.push_back(id);
        assert(parent.global_transform);
        node.global_transform = mul(*parent.global_transform, node.local_transform);
    } else {
        node.global_transform = node.local_transform;
    }
    node.global_inverse_transposed = inverse(transpose(*node.global_transform));

    scene_graph[id] = std::move(node);
    return id;
}

void Scene::add_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_ids.is_used(mesh_id));
    assert(node_ids.is_used(node_id));

    if ((!(meshes[mesh_id]->flags & MeshFlags::IsDynamic) || pretransform_dynamic) &&
        meshes[mesh_id]->instances.size() == 1) {
        // - instances was 1 and is static: previously we could pretransform, but now this is not
        // possible anymore.
        mark_mesh_dirty(mesh_id);
    }
    meshes[mesh_id]->instances.insert(node_id);
    needs_regroup = true;
}

void Scene::remove_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_ids.is_used(mesh_id));
    assert(node_ids.is_used(node_id));

    if (meshes[mesh_id]->instances.erase(node_id) > 0) {
        // pretransform may flip when instance count drops to 1 / 0
        mark_mesh_dirty(mesh_id);
        needs_regroup = true;
    }
}

void Scene::remove_mesh(const MeshID mesh_id) {
    assert(mesh_ids.is_used(mesh_id));

    meshes[mesh_id]->instances.clear();
    meshes[mesh_id].reset();

    // No cmd in scope; defer keep-alive to the next update().
    if (mesh_id < vertex_buffers.size() && vertex_buffers[mesh_id]) {
        pending_buffer_releases.push_back(std::move(vertex_buffers[mesh_id]));
    }
    if (mesh_id < prev_vertex_buffers.size() && prev_vertex_buffers[mesh_id]) {
        pending_buffer_releases.push_back(std::move(prev_vertex_buffers[mesh_id]));
    }
    if (mesh_id < index_buffers.size() && index_buffers[mesh_id]) {
        pending_buffer_releases.push_back(std::move(index_buffers[mesh_id]));
    }

    mesh_ids.release(mesh_id);
    meshes.resize(mesh_ids.size());
    vertex_buffers.resize(mesh_ids.size());
    prev_vertex_buffers.resize(mesh_ids.size());
    index_buffers.resize(mesh_ids.size());
    needs_regroup = true;
}

void Scene::remove_node(const NodeID node_id) {
    assert(node_ids.is_used(node_id));
    SceneNode& node = *scene_graph[node_id];

    // Detach from parent's children list.
    if (node.parent != NODE_ID_INVALID && node_ids.is_used(node.parent)) {
        auto& siblings = scene_graph[node.parent]->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), node_id), siblings.end());
    }

    // Cascade. Snapshot first — recursive remove mutates the vector.
    const std::vector<NodeID> children_snapshot = node.children;
    for (const NodeID child : children_snapshot) {
        remove_node(child);
    }

    // Detach from every mesh that instanced this node. O(meshes); fine for a rare op.
    for (const MeshID mesh_id : mesh_ids) {
        meshes[mesh_id]->instances.erase(node_id);
    }
    needs_regroup = true;

    scene_graph[node_id].reset();
    node_ids.release(node_id);
    scene_graph.resize(node_ids.size());
}

void Scene::mark_mesh_dirty(const MeshID mesh_id) {
    assert(mesh_ids.is_used(mesh_id));
    // assert((get_mesh(mesh_id).flags & MeshFlags::IsDynamic) == 0);
    meshes[mesh_id]->dirty = true;
}

CameraID Scene::add_camera(CameraHandle camera) {
    const CameraID id = cameras.size();
    cameras.push_back(std::move(camera));
    return id;
}

std::vector<CameraHandle> Scene::get_cameras() const {
    return cameras;
}

CameraHandle Scene::get_camera(const CameraID camera_id) const {
    assert(camera_id < cameras.size());
    return cameras[camera_id];
}

CameraHandle Scene::get_active_camera() const {
    assert(!cameras.empty());
    return cameras[active_camera];
}

void Scene::set_active_camera(const uint32_t index) {
    assert(index < cameras.size());
    active_camera = index;
}

void Scene::set_pretransform_dynamic(bool value) {
    if (pretransform_dynamic == value)
        return;
    pretransform_dynamic = value;

    // assume all dynamic meshes need reupload.
    for (const MeshID mesh_id : mesh_ids) {
        if (meshes[mesh_id]->flags & MeshFlags::IsDynamic) {
            mark_mesh_dirty(mesh_id);
        }
    }
}

void Scene::node_properties(Properties& props, const SceneNode& node) {
    props.output_text("{}", node);

    for (uint32_t i = 0; i < node.children.size(); i++) {
        const SceneNode& child = *scene_graph[node.children[i]];
        if (props.st_begin_child(fmt::format("child_{:02} ({})", i, child.name),
                                 fmt::format("Child {:02} ({})", i, child.name))) {
            Scene::node_properties(props, child);
            props.st_end_child();
        }
    }
}

void Scene::properties(Properties& props) {

    if (props.st_begin_child("scene", "Explorer")) {
        if (props.st_begin_child("cameras", "Cameras")) {
            if (!cameras.empty()) {
                props.config_uint("active", active_camera, "", 0u,
                                  static_cast<uint32_t>(cameras.size()));
                if (props.is_ui()) {
                    props.st_separate("Active Camera");
                    get_active_camera()->properties(props);
                    if (aabb.is_valid() &&
                        props.config_bool("Fit AABB",
                                          "Rotates and moves the camera to fit the scene AABB.")) {
                        get_active_camera()->look_at_bounding_box(aabb);
                    }
                }
            }
            props.st_separate("Debug");
            if (props.config_bool("debug camera", enable_debug_camera) && enable_debug_camera) {
                if (debug_camera_id == CAMERA_ID_INVALID) {
                    auto db = std::make_shared<Camera>(float3(1, 0, 0), float3(0, 0, 0), get_up(),
                                                       90.f, 1920.f / 1080.f, 0.01f, 1000.f);
                    if (aabb.is_valid()) {
                        db->look_at_bounding_box(aabb);
                    }
                    debug_camera_id = add_camera(db);
                    set_active_camera(debug_camera_id);
                }
            }
            if (enable_debug_camera) {
                if (props.config_bool("make active")) {
                    set_active_camera(debug_camera_id);
                }
                if (!props.is_ui()) {
                    get_camera(debug_camera_id)->properties(props);
                }
            }

            props.st_end_child();
        }

        if (props.st_begin_child("meshes", "Meshes")) {
            for (const MeshID id : mesh_ids) {
                if (props.st_begin_child(fmt::format("mesh_{:02} ", id),
                                         fmt::format("{:02}: {}", id, meshes[id]->name))) {
                    props.output_text("{}", *meshes[id]);
                    props.st_end_child();
                }
            }
            props.st_end_child();
        }

        if (props.st_begin_child("graph", "Graph")) {
            for (const NodeID id : node_ids) {
                const SceneNode& node = *scene_graph[id];
                if (node.parent != NODE_ID_INVALID) {
                    continue;
                }
                if (props.st_begin_child(fmt::format("node_{:02} ", id),
                                         fmt::format("{:02}: {}", id, node.name))) {
                    node_properties(props, node);
                    props.st_end_child();
                }
            }
            props.st_end_child();
        }
        props.st_end_child();
    }

    if (props.st_begin_child("settings", "Settings")) {
        bool pretransform = get_pretransform_dynamic();
        if (props.config_bool("Pretransform Dyanmic", pretransform)) {
            set_pretransform_dynamic(pretransform);
        }
        props.st_end_child();
    }

    if (props.st_begin_child("stats", "Statistics")) {
        std::size_t total_vertices = 0;
        std::size_t total_triangles = 0;
        for (const MeshID id : mesh_ids) {
            total_vertices += meshes[id]->get_vertex_count();
            total_triangles += meshes[id]->get_primitive_count();
        }

        props.output_text("nodes: {}\nmeshes: {}\nvertices: {}\ntriangles: {}\nmaterials: "
                          "{}\ntextures: {}",
                          node_ids.count(), mesh_ids.count(), total_vertices, total_triangles,
                          material_system->get_material_count(),
                          get_texture_manager()->get_texture_count());

        if (aabb.is_valid()) {
            props.output_text("aabb: min={}, max={}, size={}", aabb.get_min(), aabb.get_max(),
                              aabb.get_max() - aabb.get_min());
        } else {
            props.output_text("aabb: <not available>");
        }

        props.st_end_child();
    }
}

// ---------------------------------------------------------------------------
// Scene update
// ---------------------------------------------------------------------------

void Scene::compute_mesh_groups() {

    auto prev_mesh_groups = std::move(mesh_groups);
    auto prev_mesh_to_group = std::move(mesh_to_group);
    mesh_groups.clear();
    mesh_to_group.assign(mesh_ids.size(), MESH_GROUP_ID_INVALID);

    // allow to access prev_mesh_to_group with any (new) MeshID.
    prev_mesh_to_group.resize(mesh_ids.size(), MESH_GROUP_ID_INVALID);
    mesh_groups.reserve(std::min<std::size_t>(mesh_ids.count(), prev_mesh_groups.size()));

    // 1. Group meshes according to our grouping logic

    // See description in scene.hpp for grouping logic

    // have their global transforms precomputed
    std::unordered_map<MeshFlags, MeshGroupID> mesh_groups_static_non_instanced;
    // have their global transforms precomputed only if enabled
    // packed (GeometryFlags << 32 | NodeID) -> MeshGroupID
    std::unordered_map<uint64_t, MeshGroupID> mesh_groups_dynamic_non_instanced;
    // CANNOT have their global transforms precomputed
    std::map<std::set<NodeID>, std::unordered_map<MeshFlags, MeshGroupID>> mesh_groups_instanced;

    mesh_groups_static_non_instanced.reserve(32 /*all flags set*/);
    mesh_groups_dynamic_non_instanced.reserve(node_ids.count() * 2);

    for (const MeshID mesh_id : mesh_ids) {
        const Mesh& mesh = *meshes[mesh_id];
        MeshGroupID group_id = MESH_GROUP_ID_INVALID;

        if (mesh.instances.empty()) {
            // dont build BLASes for meshes that are not used.
            continue;
        }

        if (mesh.instances.size() > 1) {
            // is instanced
            auto& by_instances = mesh_groups_instanced[mesh.instances];
            const auto [it, inserted] = by_instances.try_emplace(mesh.flags, mesh_groups.size());
            if (inserted) {
                mesh_groups.emplace_back();
            }
            group_id = it->second;
        } else {
            if (mesh.flags & MeshFlags::IsDynamic) {
                // dynamic, non-instanced
                assert(mesh.instances.size() == 1);
                const uint64_t key =
                    (static_cast<uint64_t>(mesh.flags) << 32) | (*mesh.instances.begin());
                const auto [it, inserted] =
                    mesh_groups_dynamic_non_instanced.try_emplace(key, mesh_groups.size());
                if (inserted) {
                    mesh_groups.emplace_back();
                }
                group_id = it->second;
            } else {
                // static, non-instanced
                assert(mesh.instances.size() == 1);
                const auto [it, inserted] =
                    mesh_groups_static_non_instanced.try_emplace(mesh.flags, mesh_groups.size());
                if (inserted) {
                    mesh_groups.emplace_back();
                }
                group_id = it->second;
            }
        }

        assert(group_id != MESH_GROUP_ID_INVALID);
        mesh_to_group[mesh_id] = group_id;
        MeshGroup& group = mesh_groups[group_id];
        assert(group.meshes.empty() || group.flags == mesh.flags);
        group.flags = mesh.flags;
        group.meshes.insert(mesh_id);
    }

    // 2. Now go over the groups and try to find previous BLASs for the groups to prevent
    // (re)builds.
    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        // We can check the first mesh only, since if thats not in the old group we cant reuse the
        // group / BLAS anyway.
        MeshGroupID maybe_prev_group = prev_mesh_to_group[*group.meshes.begin()];
        if (maybe_prev_group != MESH_GROUP_ID_INVALID) {
            MeshGroup& prev_group = prev_mesh_groups[maybe_prev_group];
            if (prev_group.meshes == group.meshes) {
                group.blas = prev_group.blas;
                // blas_dirty is computed later when we upload the meshes, because this method is
                // not run every frame.We later also check if the blas can be actually reused or if
                // a new one is necessary.
            }
        }
    }
}

void Scene::upload_geometry_data_and_transforms(const CommandBufferHandle& cmd) {
    geometries.clear();
    instance_transforms.clear();
    inverse_transposed_instance_transforms.clear();
    prev_instance_transforms.clear();
    prev_inverse_transposed_instance_transforms.clear();

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        const bool pretransform = group.is_pretranformed(meshes, pretransform_dynamic);

        for (const NodeID node_id : group.get_instances(meshes)) {
            for (const MeshID mesh_id : group.meshes) {
                Mesh& mesh = *meshes[mesh_id];
                assert(vertex_buffers[mesh_id] && index_buffers[mesh_id]);

                GeometryData gd;
                gd.material_id = mesh.material_id;
                gd.vertices = vertex_buffers[mesh_id]->get_device_address();
                gd.indices = index_buffers[mesh_id]->get_device_address();
                gd.prev_vertices = prev_vertex_buffers[mesh_id]
                                       ? prev_vertex_buffers[mesh_id]->get_device_address()
                                       : vk::DeviceAddress{0};
                gd.flags = pretransform ? GeometryDataFlags::Pretransformed : GeometryDataFlags{};

                geometries.emplace_back(gd);
            }

            if (pretransform) {
                const float4x4 identity_transform = identity();

                instance_transforms.emplace_back(identity_transform);
                inverse_transposed_instance_transforms.emplace_back(identity_transform);
                prev_instance_transforms.emplace_back(identity_transform);
                prev_inverse_transposed_instance_transforms.emplace_back(identity_transform);
            } else {
                SceneNode& node = *scene_graph[node_id];

                instance_transforms.emplace_back(node.global_transform.value());
                inverse_transposed_instance_transforms.emplace_back(
                    node.global_inverse_transposed.value());

                // TODO: Update here when animating the transforms is supported!
                prev_instance_transforms.emplace_back(node.global_transform.value());
                prev_inverse_transposed_instance_transforms.emplace_back(
                    node.global_inverse_transposed.value());
            }
        }
    }

    const auto staging = allocator->get_staging();
    auto c = shader_object->get_cursor();

    if (!geometries.empty()) {

        const vk::DeviceSize geometries_size = geometries.size() * sizeof(GeometryData);
        if (!geometries_buffer || geometries_buffer->get_size() < geometries_size) {
            geometries_buffer = allocator->create_buffer(
                geometries.size() * sizeof(GeometryData),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                MemoryMappingType::NONE, "Scene::geometries");

            c["geometries"] = geometries_buffer;
        }
        staging->cmd_to_device(cmd, geometries_buffer, geometries);
    }

    if (!instance_transforms.empty()) {
        const vk::DeviceSize transforms_size = instance_transforms.size() * sizeof(float4x4);

        if (!instance_transforms_buffer ||
            instance_transforms_buffer->get_size() < transforms_size) {

            const auto transform_usage =
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
            instance_transforms_buffer =
                allocator->create_buffer(transforms_size, transform_usage, MemoryMappingType::NONE,
                                         "Scene::instance_transforms");
            inverse_transposed_instance_transforms_buffer =
                allocator->create_buffer(transforms_size, transform_usage, MemoryMappingType::NONE,
                                         "Scene::inv_transposed_instance_transforms");
            prev_instance_transforms_buffer =
                allocator->create_buffer(transforms_size, transform_usage, MemoryMappingType::NONE,
                                         "Scene::prev_instance_transforms");
            prev_inverse_transposed_instance_transforms_buffer =
                allocator->create_buffer(transforms_size, transform_usage, MemoryMappingType::NONE,
                                         "Scene::prev_inv_transposed_instance_transforms");

            c["instance_transforms"] = instance_transforms_buffer;
            c["inverse_transposed_instance_transforms"] =
                inverse_transposed_instance_transforms_buffer;
            c["prev_instance_transforms"] = prev_instance_transforms_buffer;
            c["prev_inverse_transposed_instance_transforms"] =
                prev_inverse_transposed_instance_transforms_buffer;
        }

        staging->cmd_to_device(cmd, instance_transforms_buffer, instance_transforms);
        staging->cmd_to_device(cmd, inverse_transposed_instance_transforms_buffer,
                               inverse_transposed_instance_transforms);
        staging->cmd_to_device(cmd, prev_instance_transforms_buffer, prev_instance_transforms);
        staging->cmd_to_device(cmd, prev_inverse_transposed_instance_transforms_buffer,
                               prev_inverse_transposed_instance_transforms);
    }
}

void Scene::upload_meshes(const CommandBufferHandle& cmd) {
    // For now we have a index and vertex buffer for each mesh,
    // we could combine them per group or into single buffers in the future.
    if (mesh_ids.size() > vertex_buffers.size()) {
        index_buffers.resize(mesh_ids.size());
        vertex_buffers.resize(mesh_ids.size());
        prev_vertex_buffers.resize(mesh_ids.size());
    }

    const auto staging = allocator->get_staging();
    const auto buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer |
                              vk::BufferUsageFlagBits::eTransferDst |
                              vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                              vk::BufferUsageFlagBits::eShaderDeviceAddress;

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        const bool pretransform_group = group.is_pretranformed(meshes, pretransform_dynamic);

        for (const MeshID mesh_id : group.meshes) {
            Mesh& mesh = *meshes[mesh_id];
            group.blas_dirty |= mesh.dirty;

            if (mesh.dirty) {
                assert(!mesh.instances.empty());

                const uint32_t vertex_count = mesh.get_vertex_count();
                const uint32_t primitive_count = mesh.get_primitive_count();
                const vk::DeviceSize vb_size = vertex_count * sizeof(PackedVertexData);
                const vk::DeviceSize prev_vb_size = vertex_count * sizeof(PackedPrevVertexData);
                const vk::DeviceSize ib_size = primitive_count * sizeof(uint3);

                // Allocate / Ensure (prev)vertex and index buffers.
                // Previous frames reference these via raw device addresses in their
                // geometries buffer, so replaced handles must outlive in-flight work.
                assert(mesh_id < vertex_buffers.size());
                auto& vb = vertex_buffers[mesh_id];
                if (!vb || (vb->get_size() < vb_size)) {
                    if (vb)
                        cmd->keep_until_pool_reset(std::move(vb));
                    vb = allocator->create_buffer(
                        vb_size, buffer_usage, MemoryMappingType::NONE,
                        fmt::format("Scene::vb[{}]: name={}", mesh_id, mesh.name));
                }

                // Only allocate prev vertices for dynamic meshes.
                assert(mesh_id < prev_vertex_buffers.size());
                auto& prev_vb = prev_vertex_buffers[mesh_id];
                if (mesh.is_dynamic()) {
                    if (!prev_vb || (prev_vb->get_size() < prev_vb_size)) {
                        if (prev_vb)
                            cmd->keep_until_pool_reset(std::move(prev_vb));
                        prev_vb = allocator->create_buffer(
                            prev_vb_size, buffer_usage, MemoryMappingType::NONE,
                            fmt::format("Scene::prev_vb[{}]: name={}", mesh_id, mesh.name));
                    }
                } else if (prev_vb) {
                    cmd->keep_until_pool_reset(std::move(prev_vb));
                }

                assert(mesh_id < index_buffers.size());
                auto& ib = index_buffers[mesh_id];
                if (!ib || (ib->get_size() < ib_size)) {
                    if (ib)
                        cmd->keep_until_pool_reset(std::move(ib));
                    ib = allocator->create_buffer(
                        ib_size, buffer_usage, MemoryMappingType::NONE,
                        fmt::format("Scene::ib[{}]: name={}", mesh_id, mesh.name));
                }

                // Upload vertices
                const NodeID node_id = *mesh.instances.begin();
                const SceneNode& node = *scene_graph[node_id];
                assert(node.global_transform);
                // dont pretransform if identity to save some computations
                const bool pretransform_mesh =
                    pretransform_group && node.global_transform.value() != identity();

                const MemoryAllocationHandle vb_staging = staging->cmd_to_device(cmd, vb);
                auto* vb_mapped = vb_staging->map_as<PackedVertexData>();
                if (pretransform_mesh) {
                    assert(mesh.instances.size() == 1 && "for pretransformed meshes the transform "
                                                         "must be unique, i.e. there is only one!");
                    for (uint32_t v = 0; v < vertex_count; v++) {
                        vb_mapped[v] = mesh.get_packed_vertex_pretransformed(v, node);
                    }
                } else {
                    for (uint32_t v = 0; v < vertex_count; v++) {
                        vb_mapped[v] = mesh.get_packed_vertex(v);
                    }
                }
                vb_staging->unmap();

                // Upload prev vertices (dynamic meshes only).
                if (prev_vb) {
                    const MemoryAllocationHandle prev_vtx_staging =
                        staging->cmd_to_device(cmd, prev_vb);
                    auto* prev_vb_mapped = prev_vtx_staging->map_as<PackedPrevVertexData>();
                    if (pretransform_mesh) {
                        assert(mesh.instances.size() == 1 &&
                               "for pretransformed meshes the transform "
                               "must be unique, i.e. there is only one!");
                        for (uint32_t v = 0; v < vertex_count; v++) {
                            prev_vb_mapped[v] = mesh.get_packed_prev_vertex_pretransformed(v, node);
                        }
                    } else {
                        for (uint32_t v = 0; v < vertex_count; v++) {
                            prev_vb_mapped[v] = mesh.get_packed_prev_vertex(v);
                        }
                    }
                    prev_vtx_staging->unmap();
                }

                // Upload indices
                const MemoryAllocationHandle ib_staging = staging->cmd_to_device(cmd, ib);
                auto* ib_mapped = ib_staging->map_as<uint3>();
                for (uint32_t p = 0; p < primitive_count; p++) {
                    ib_mapped[p] = mesh.get_indices(p);
                }
                ib_staging->unmap();

                mesh.dirty = false;
            }
        }
    }
}

void Scene::build_blas(const CommandBufferHandle& cmd) {
    //     if (mesh_groups.empty())
    //         return;

    blas_geometries.assign(mesh_groups.size(), {});

    bool did_build_static = false;

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        if (group.blas && !group.blas_dirty) {
            continue;
        }

        auto& blas_geometry = blas_geometries[group_id];

        for (MeshID mesh_id : group.meshes) {
            const Mesh& mesh = *meshes[mesh_id];

            vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
            triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
            triangles.vertexData = vertex_buffers[mesh_id]->get_device_address();
            triangles.vertexStride = sizeof(PackedVertexData);
            triangles.maxVertex = mesh.get_vertex_count() - 1;
            triangles.indexType = vk::IndexType::eUint32;
            triangles.indexData = index_buffers[mesh_id]->get_device_address();

            vk::AccelerationStructureGeometryKHR geom;
            geom.geometryType = vk::GeometryTypeKHR::eTriangles;
            geom.geometry.triangles = triangles;

            if (mesh.flags & IsOpaque) {
                geom.flags = vk::GeometryFlagBitsKHR::eOpaque;
            }

            vk::AccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = mesh.get_primitive_count();

            blas_geometry.geometries.push_back(geom);
            blas_geometry.ranges.push_back(range);
        }

        const bool is_static = !(group.flags & IsDynamic);
        vk::BuildAccelerationStructureFlagsKHR flags;
        if (is_static) {
            did_build_static = true;
            flags |= vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
        }

        const auto size_info =
            as_builder.get_size_info(blas_geometry.geometries, blas_geometry.ranges, flags);

        if (!group.blas || group.blas->get_size() < size_info.accelerationStructureSize) {
            // cannot reuse and needs to be allcated
            group.blas = allocator->create_acceleration_structure(
                vk::AccelerationStructureTypeKHR::eBottomLevel, size_info,
                fmt::format("Scene::blas[{}]", group_id));
        }

        as_builder.queue_build(blas_geometry.geometries, blas_geometry.ranges, group.blas,
                               size_info, flags);
        group.blas_dirty = false;
    }

    as_builder.get_cmds_blas(cmd, as_scratch_buffer);

    if (did_build_static) {
        // release a probably large scratch buffer
        as_scratch_buffer.reset();
    }
}

void Scene::build_tlas(const CommandBufferHandle& cmd) {

    tlas_instances.clear();
    // we set this such that GeometryData index is InstanceID + GeometryIndex.
    uint32_t instance_id = 0;

    for (uint32_t group_id = 0; group_id < mesh_groups.size(); group_id++) {
        // iterate in the same way as upload_geometry_data_and_transforms!

        const auto& group = mesh_groups[group_id];
        assert(group.blas);

        for (NodeID node_id : group.get_instances(meshes)) {
            vk::AccelerationStructureInstanceKHR tlas_instance{};

            tlas_instance.instanceCustomIndex = instance_id;
            tlas_instance.mask = 0xFF;
            tlas_instance.accelerationStructureReference =
                group.blas->get_acceleration_structure_device_address();

            if (group.is_pretranformed(meshes, pretransform_dynamic)) {
                // Use identity
                tlas_instance.transform.matrix[0][0] = 1.f;
                tlas_instance.transform.matrix[1][1] = 1.f;
                tlas_instance.transform.matrix[2][2] = 1.f;
            } else {
                const auto& t = *scene_graph[node_id]->global_transform;
                for (int row = 0; row < 3; row++)
                    for (int col = 0; col < 4; col++)
                        tlas_instance.transform.matrix[row][col] = t[row][col];
            }

            vk::GeometryInstanceFlagsKHR geometry_instance_flags{};
            if (group.flags & IsOpaque) {
                // we group by flags, and thus can force the whole group to opaque.
                // TODO: Maybe reduce groups by only setting this in geometry level?
                geometry_instance_flags |= vk::GeometryInstanceFlagBitsKHR::eForceOpaque;
            }
            if (group.flags & FrontCounterClockwise) {
                geometry_instance_flags |=
                    vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise;
            }
            if (group.flags & TwoSided) {
                geometry_instance_flags |=
                    vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable;
            }
            tlas_instance.setFlags(geometry_instance_flags);

            tlas_instances.emplace_back(tlas_instance);
            instance_id += static_cast<uint32_t>(group.meshes.size());
        }
    }

    // empty buffers are not allowed, but we need a buffer to build an empty TLAS.
    const vk::DeviceSize tlas_instances_size = std::max((std::size_t)1, tlas_instances.size()) *
                                               sizeof(vk::AccelerationStructureInstanceKHR);
    if (!tlas_instances_buffer || tlas_instances_buffer->get_size() < tlas_instances_size) {
        tlas_instances_buffer = allocator->create_buffer(
            tlas_instances_size,
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            MemoryMappingType::NONE, "Scene::tlas_instances");
    }

    allocator->get_staging()->cmd_to_device(cmd, tlas_instances_buffer, tlas_instances);

    cmd->barrier(tlas_instances_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead));

    const auto size_info =
        as_builder.get_size_info(tlas_instances.size(), tlas_instances_buffer,
                                 vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
    if (!tlas || tlas->get_size() < size_info.accelerationStructureSize) {
        // cannot reuse and needs to be allcated
        tlas = allocator->create_acceleration_structure(vk::AccelerationStructureTypeKHR::eTopLevel,
                                                        size_info);
        shader_object->get_cursor()["as"]["as"] = tlas;
    }

    as_builder.queue_build(tlas_instances.size(), tlas_instances_buffer, tlas, size_info);
    as_builder.get_cmds_tlas(cmd, as_scratch_buffer);
}

void Scene::update(const CommandBufferHandle& cmd,
                   const float time,
                   const float time_diff,
                   const uint32_t frame) {
    on_update(cmd, time, time_diff, frame);

    assert(!cameras.empty() &&
           "the scene implementation must ensure that there is at least one camera");

    for (auto& buffer : pending_buffer_releases) {
        cmd->keep_until_pool_reset(std::move(buffer));
    }
    pending_buffer_releases.clear();

    material_system->update(cmd);

    // do that before to upload geometry buffers, because that clears the dirty flags!
    if (needs_regroup) {
        compute_mesh_groups();

        needs_regroup = false;
    }

    // Allocate/upload vertex/index buffers first so their device addresses are available
    // when GeometryData is written below.
    upload_meshes(cmd);

    // needs to be moved once nodes support animating the transforms
    upload_geometry_data_and_transforms(cmd);

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    });

    if (true /*build_as*/) {
        build_blas(cmd);
        // TODO only build if blases where built or instances where modified.
        build_tlas(cmd);
        // TODO: compact the static BLASes -> reduce memory bandwidth and increase performance.

        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer |
                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eTransferWrite |
                vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
        });
    }

    auto c = shader_object->get_cursor();

    c["frame"] = frame;
    c["time"] = get_time(time);
    c["time_diff"] = time_diff;

    const auto cam = get_active_camera();
    assert(cam);

    prev_active_camera.write_to(c["prev_camera"]);
    cam->write_to(c["camera"]);
    prev_active_camera = *cam;
}

} // namespace merian
