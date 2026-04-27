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

namespace {

// HostPacked pretransform: unpack, transform, repack a single vertex.
PackedVertexData
pretransform_packed_vertex(const PackedVertexData& src, const float4x4& m, const float4x4& m_it) {
    const float3 normal = decode_normal(src.encoded_normal);
    const float3 t_dir = decode_normal(src.encoded_tangent & ~1u);
    const float t_sign = (src.encoded_tangent & 1u) ? -1.f : 1.f;

    PackedVertexData v;
    v.position = mul(m, float4(src.position, 1.f));
    v.encoded_normal = encode_normal(normalize(float3(mul(m_it, float4(normal, 0.f)))));
    v.uv = src.uv;
    const float3 tw = normalize(float3(mul(m_it, float4(t_dir, 0.f))));
    v.encoded_tangent = pack_tangent(tw, t_sign);
    return v;
}

PackedPrevVertexData pretransform_packed_prev_vertex(const PackedPrevVertexData& src,
                                                     const float4x4& m) {
    return PackedPrevVertexData{.position = mul(m, float4(src.position, 1.f))};
}

// in bytes
vk::DeviceSize size_for_index_type(const vk::IndexType type) {
    switch (type) {

    case vk::IndexType::eUint16:
        return 2;
    case vk::IndexType::eUint32:
        return 4;
    case vk::IndexType::eUint8:
        return 1;
    case vk::IndexType::eNoneKHR:
        assert(false && "not yet supported!");
    }
}

} // namespace

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

    composition = SlangComposition::create();
    composition->add_composition(material_system->get_composition());
    composition->add_module_from_path("merian-shaders/scene/scene.slang");
    composition->add_module_from_path("merian-shaders/scene/camera.slang");
    composition->add_module_from_path("merian-shaders/scene/environment-map.slang");
    composition->add_module_from_path("merian-shaders/scene/acceleration-structure.slang");

    layout_program = SlangProgram::create(compile_context, composition);
    layout_program->on_changed(layout_program, [&] { rebuild_shader_object(); });

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
    SPDLOG_DEBUG("recreate shader object");

    shader_object = layout_program->create_shader_object(context, "merian::Scene", obj_allocator);

    auto c = shader_object->get_cursor();

    c["material_system"] = material_system;
    material_system->on_changed(
        shader_object, [&] { shader_object->get_cursor()["material_system"] = material_system; });

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

    return id;
}

NodeID Scene::add_node(SceneNode node) {
    const NodeID id = node_ids.acquire();

    if (id >= scene_graph.size()) {
        scene_graph.resize(node_ids.size());
    }

    if (node.parent != NODE_ID_INVALID) {
        assert(node_ids.is_used(node.parent));
        assert(scene_graph[node.parent]);
        scene_graph[node.parent]->children.push_back(id);
    }

    scene_graph[id] = std::move(node);
    return id;
}

void Scene::invalidate_node(SceneNode& node) {
    node.global_transform.reset();
    node.global_inverse_transposed.reset();

    for (const auto child_node_id : node.children) {
        assert(node_ids.is_used(child_node_id));
        assert(child_node_id < scene_graph.size());
        assert(scene_graph[child_node_id]);

        invalidate_node(*scene_graph[child_node_id]);
    }
}

void Scene::update_node(NodeID node_id, const float4x4& local_transform) {
    assert(node_ids.is_used(node_id));
    assert(node_id < scene_graph.size());
    assert(scene_graph[node_id]);

    SceneNode& node = *scene_graph[node_id];
    node.local_transform = local_transform;

    invalidate_node(node);

    transforms_changed = true;
}

void Scene::add_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_ids.is_used(mesh_id));
    assert(node_ids.is_used(node_id));

    Mesh& mesh = *meshes[mesh_id];

    if (mesh.instances.size() == 1 && (mesh.is_static() || pretransform_dynamic)) {
        // - instances was 1 and is static: previously we could pretransform, but now this is not
        // possible anymore.
        mesh.vertices_dirty = true;
    }
    mesh.instances.insert(node_id);
    needs_regroup = true;
}

void Scene::remove_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_ids.is_used(mesh_id));
    assert(node_ids.is_used(node_id));

    Mesh& mesh = *meshes[mesh_id];

    if (mesh.instances.erase(node_id) > 0) {
        if (mesh.instances.size() == 1 && (mesh.is_static() || pretransform_dynamic)) {
            // instances droped to 1 and is static: previously we could not pretransform, not this
            // is possible
            mesh.vertices_dirty = true;
        }
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

    // Remove from parent's children list.
    if (node.parent != NODE_ID_INVALID) {
        assert(node_ids.is_used(node.parent));

        auto& siblings = scene_graph[node.parent]->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), node_id), siblings.end());
    }

    while (!node.children.empty()) {
        // recursive remove mutates the vector.
        remove_node(node.children.back());
    }

    // Remove all instances on this node.
    for (const MeshID mesh_id : mesh_ids) {
        assert(meshes[mesh_id]);
        meshes[mesh_id]->instances.erase(node_id);
    }

    needs_regroup = true;

    scene_graph[node_id].reset();
    node_ids.release(node_id);
    scene_graph.resize(node_ids.size());
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
            meshes[mesh_id]->vertices_dirty = true;
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

void Scene::upload_transforms(const CommandBufferHandle& cmd) {
    instance_transforms.clear();
    inverse_transposed_instance_transforms.clear();
    prev_instance_transforms.clear();
    prev_inverse_transposed_instance_transforms.clear();

    const float4x4 identity_transform = identity();

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        const bool pretransform = group.is_pretranformed(meshes, pretransform_dynamic);

        for (const NodeID node_id : group.get_instances(meshes)) {
            SceneNode& node = *scene_graph[node_id];
            const float4x4& global_transform = get_global_transform(node);
            const float4x4& global_inverse_transposed_transform =
                get_global_inverse_transposed_transform(node);

            if (pretransform) {
                instance_transforms.emplace_back(identity_transform);
                inverse_transposed_instance_transforms.emplace_back(identity_transform);
                prev_instance_transforms.emplace_back(identity_transform);
                prev_inverse_transposed_instance_transforms.emplace_back(identity_transform);
            } else {
                instance_transforms.emplace_back(global_transform);
                inverse_transposed_instance_transforms.emplace_back(
                    global_inverse_transposed_transform);

                prev_instance_transforms.emplace_back(
                    node.prev_global_transform.value_or(global_transform));
                prev_inverse_transposed_instance_transforms.emplace_back(
                    node.prev_global_inverse_transposed.value_or(
                        global_inverse_transposed_transform));
            }
        }
    }

    const auto staging = allocator->get_staging();
    auto c = shader_object->get_cursor();

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

    for (const NodeID& node_id : node_ids) {
        assert(scene_graph[node_id]);
        SceneNode& node = *scene_graph[node_id];

        node.prev_global_transform = node.global_transform;
        node.prev_global_inverse_transposed = node.global_inverse_transposed;
    }
}

void Scene::upload_geometry_data(const CommandBufferHandle& cmd) {
    geometries.clear();

    const float4x4 identity_transform = identity();

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        const bool pretransform = group.is_pretranformed(meshes, pretransform_dynamic);

        for (const NodeID node_id : group.get_instances(meshes)) {
            SceneNode& node = *scene_graph[node_id];
            const bool transform_is_identity = get_global_transform(node) == identity_transform;

            for (const MeshID mesh_id : group.meshes) {
                Mesh& mesh = *meshes[mesh_id];
                assert(vertex_buffers[mesh_id] && index_buffers[mesh_id]);

                GeometryData gd;
                gd.material_id = mesh.material_id;
                gd.vertices = vertex_buffers[mesh_id]->get_device_address();
                gd.indices = index_buffers[mesh_id]->get_device_address();
                assert(!(group.flags & MeshFlags::IsDynamic) || prev_vertex_buffers[mesh_id]);
                gd.prev_vertices = prev_vertex_buffers[mesh_id]
                                       ? prev_vertex_buffers[mesh_id]->get_device_address()
                                       : vk::DeviceAddress{0};
                gd.flags = GeometryDataFlags{};
                if (pretransform || transform_is_identity) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::Pretransformed);
                }
                if (group.flags & MeshFlags::IsDynamic) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::IsDynamic);
                }

                geometries.emplace_back(gd);
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
}

void Scene::upload_meshes(const CommandBufferHandle& cmd) {
    // For now we have a index and vertex buffer for each mesh,
    // we could combine them per group or into single buffers in the future.

    index_buffers.resize(mesh_ids.size());
    vertex_buffers.resize(mesh_ids.size());
    prev_vertex_buffers.resize(mesh_ids.size());

    const auto staging = allocator->get_staging();

    const auto buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer |
                              vk::BufferUsageFlagBits::eTransferDst |
                              vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                              vk::BufferUsageFlagBits::eShaderDeviceAddress;
    const auto ensure_buffer = [&](BufferHandle& slot, vk::DeviceSize size,
                                   const std::string& name) {
        if (!slot || slot->get_size() < size) {
            if (slot) {
                // make sure we keep the buffer until it is not inflight.
                cmd->keep_until_pool_reset(std::move(slot));
            }
            slot = allocator->create_buffer(size, buffer_usage, MemoryMappingType::NONE, name);
        }
    };

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        const bool pretransform_group = group.is_pretranformed(meshes, pretransform_dynamic);

        for (const MeshID mesh_id : group.meshes) {
            Mesh& mesh = *meshes[mesh_id];
            const bool mesh_dirty = mesh.is_dirty();
            if (!mesh_dirty)
                continue;

            group.blas_dirty |= mesh_dirty;

            assert(!mesh.instances.empty());
            assert(mesh_id < vertex_buffers.size());
            assert(mesh_id < prev_vertex_buffers.size());
            assert(mesh_id < index_buffers.size());

            const uint32_t vertex_count = mesh.get_vertex_count();
            const uint32_t primitive_count = mesh.get_primitive_count();
            const vk::DeviceSize vb_size = vertex_count * sizeof(PackedVertexData);
            const vk::DeviceSize prev_vb_size = vertex_count * sizeof(PackedPrevVertexData);
            const vk::DeviceSize ib_size =
                primitive_count * std::size_t(3) * size_for_index_type(mesh.index_type);

            const NodeID node_id = *mesh.instances.begin();
            const float4x4& global_transform = get_global_transform(node_id);
            const float4x4& global_inverse_transposed_transform =
                get_global_inverse_transposed_transform(node_id);

            const bool pretransform_mesh = pretransform_group && global_transform != identity();

            // TODO fix with something like:

            // template <typename VariantType, typename GeneratorFunc>
            // BufferHandle upload_buffer(const VariantType& variant, size_t byte_size,
            //                               MeshUploader& uploader, GeneratorFunc&&
            //                               generator_logic) {
            //     return std::visit(
            //         [&](auto&& source) -> BufferHandle {
            //             using T = std::decay_t<decltype(source)>;

            //             if constexpr (is_device_local<T>::value) {
            //                 return uploader.keep_local(source.data);
            //             } else if constexpr (is_device_staged<T>::value) {
            //                 // TODO: If your GPU pretransform logic goes here, pass a flag to
            //                 // uploader
            //                 return uploader.copy_staged_to_local(source.data, source.offset,
            //                                                      byte_size);
            //             } else if constexpr (is_host_packed<T>::value) {
            //                 return uploader.upload_host_bytes(source.data, byte_size);
            //             } else {
            //                 // It's a HostVirtual source (HostVertices or HostPrevVertices)
            //                 return uploader.generate_and_upload(
            //                     byte_size, [&](void* mapped_memory) {
            //                         // Pass the specific host source and the mapped memory to the
            //                         // caller's lambda
            //                         generator_logic(source, mapped_memory);
            //                     });
            //             }
            //         },
            //         variant);
            // }

            const Mesh::MeshData view = mesh.get_data();
            std::visit(
                [&](auto&& m) {
                    using T = std::decay_t<decltype(m)>;

                    auto& vb = vertex_buffers[mesh_id];
                    auto& prev_vb = prev_vertex_buffers[mesh_id];
                    auto& ib = index_buffers[mesh_id];

                    if constexpr (std::is_same_v<T, Mesh::HostPacked>) {
                        ensure_buffer(vb, vb_size,
                                      fmt::format("Scene::vb[{}]: name={}", mesh_id, mesh.name));
                        ensure_buffer(ib, ib_size,
                                      fmt::format("Scene::ib[{}]: name={}", mesh_id, mesh.name));

                        const MemoryAllocationHandle vb_st = staging->cmd_to_device(cmd, vb);
                        auto* vb_mapped = vb_st->map_as<PackedVertexData>();
                        if (pretransform_mesh) {
                            assert(mesh.instances.size() == 1);
                            for (uint32_t v = 0; v < vertex_count; v++)
                                vb_mapped[v] =
                                    pretransform_packed_vertex(m.vertices[v], global_transform,
                                                               global_inverse_transposed_transform);
                        } else {
                            std::memcpy(vb_mapped, m.vertices, vb_size);
                        }
                        vb_st->unmap();

                        if (m.prev_vertices != nullptr) {
                            ensure_buffer(
                                prev_vb, prev_vb_size,
                                fmt::format("Scene::prev_vb[{}]: name={}", mesh_id, mesh.name));
                            const MemoryAllocationHandle prev_st =
                                staging->cmd_to_device(cmd, prev_vb);
                            auto* prev_mapped = prev_st->map_as<PackedPrevVertexData>();
                            if (pretransform_mesh) {
                                for (uint32_t v = 0; v < vertex_count; v++)
                                    prev_mapped[v] = pretransform_packed_prev_vertex(
                                        m.prev_vertices[v], global_transform);
                            } else {
                                std::memcpy(prev_mapped, m.prev_vertices, prev_vb_size);
                            }
                            prev_st->unmap();
                        } else if (prev_vb) {
                            cmd->keep_until_pool_reset(std::move(prev_vb));
                        }

                        const MemoryAllocationHandle ib_st = staging->cmd_to_device(cmd, ib);
                        std::memcpy(ib_st->map_as<uint3>(), m.indices, ib_size);
                        ib_st->unmap();
                    } else if constexpr (std::is_same_v<T, Mesh::Host>) {
                        ensure_buffer(vb, vb_size,
                                      fmt::format("Scene::vb[{}]: name={}", mesh_id, mesh.name));
                        ensure_buffer(ib, ib_size,
                                      fmt::format("Scene::ib[{}]: name={}", mesh_id, mesh.name));

                        const MemoryAllocationHandle vb_st = staging->cmd_to_device(cmd, vb);
                        auto* vb_mapped = vb_st->map_as<PackedVertexData>();
                        if (pretransform_mesh) {
                            assert(mesh.instances.size() == 1);
                            for (uint32_t v = 0; v < vertex_count; v++)
                                m->write_packed_vertex_pretransformed(
                                    v, global_transform, global_inverse_transposed_transform,
                                    &vb_mapped[v]);
                        } else {
                            for (uint32_t v = 0; v < vertex_count; v++)
                                m->write_packed_vertex(v, &vb_mapped[v]);
                        }
                        vb_st->unmap();

                        if (mesh.is_dynamic()) {
                            ensure_buffer(
                                prev_vb, prev_vb_size,
                                fmt::format("Scene::prev_vb[{}]: name={}", mesh_id, mesh.name));
                            const MemoryAllocationHandle prev_st =
                                staging->cmd_to_device(cmd, prev_vb);
                            auto* prev_mapped = prev_st->map_as<PackedPrevVertexData>();
                            if (pretransform_mesh) {
                                for (uint32_t v = 0; v < vertex_count; v++)
                                    m->write_packed_prev_vertex_pretransformed(v, global_transform,
                                                                               &prev_mapped[v]);
                            } else {
                                for (uint32_t v = 0; v < vertex_count; v++)
                                    m->write_packed_prev_vertex(v, &prev_mapped[v]);
                            }
                            prev_st->unmap();
                        } else if (prev_vb) {
                            cmd->keep_until_pool_reset(std::move(prev_vb));
                        }

                        const MemoryAllocationHandle ib_st = staging->cmd_to_device(cmd, ib);
                        auto* ib_mapped = ib_st->map_as<uint3>();
                        for (uint32_t p = 0; p < primitive_count; p++)
                            m->write_index(p, &ib_mapped[p]);
                        ib_st->unmap();
                    } else if constexpr (std::is_same_v<T, Mesh::DeviceStaged>) {
                        assert(!pretransform_mesh);

                        ensure_buffer(vb, vb_size,
                                      fmt::format("Scene::vb[{}]: name={}", mesh_id, mesh.name));
                        ensure_buffer(ib, ib_size,
                                      fmt::format("Scene::ib[{}]: name={}", mesh_id, mesh.name));

                        cmd->copy(m.vertex_buffer, vb,
                                  {vk::BufferCopy{m.vertex_offset, 0, vb_size}});

                        if (mesh.is_dynamic()) {
                            assert(m.prev_vertex_buffer);
                            ensure_buffer(
                                prev_vb, prev_vb_size,
                                fmt::format("Scene::prev_vb[{}]: name={}", mesh_id, mesh.name));
                            cmd->copy(m.prev_vertex_buffer, prev_vb,
                                      {vk::BufferCopy{m.prev_vertex_offset, 0, prev_vb_size}});
                        } else if (prev_vb) {
                            cmd->keep_until_pool_reset(std::move(prev_vb));
                        }

                        cmd->copy(m.index_buffer, ib, {vk::BufferCopy{m.index_offset, 0, ib_size}});
                    } else { // DeviceLocal
                        static_assert(std::is_same_v<T, Mesh::DeviceLocal>);
                        assert(!pretransform_mesh);

                        if (vb && vb != m.vertex_buffer) {
                            cmd->keep_until_pool_reset(std::move(vb));
                        }
                        vb = m.vertex_buffer;

                        if (mesh.is_dynamic()) {
                            assert(m.prev_vertex_buffer);
                            if (prev_vb && prev_vb != m.prev_vertex_buffer)
                                cmd->keep_until_pool_reset(std::move(prev_vb));
                            prev_vb = m.prev_vertex_buffer;
                        } else if (prev_vb) {
                            cmd->keep_until_pool_reset(std::move(prev_vb));
                        }

                        if (ib && ib != m.index_buffer)
                            cmd->keep_until_pool_reset(std::move(ib));
                        ib = m.index_buffer;
                    }
                },
                view);

            mesh.dirty = false;
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
            triangles.indexType = mesh.index_type;
            triangles.indexData = index_buffers[mesh_id]->get_device_address();

            vk::AccelerationStructureGeometryKHR geom;
            geom.geometryType = vk::GeometryTypeKHR::eTriangles;
            geom.geometry.triangles = triangles;

            if (mesh.flags & MeshFlags::IsOpaque) {
                geom.flags = vk::GeometryFlagBitsKHR::eOpaque;
            }

            vk::AccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = mesh.get_primitive_count();

            blas_geometry.geometries.push_back(geom);
            blas_geometry.ranges.push_back(range);
        }

        const bool is_static = !(group.flags & MeshFlags::IsDynamic);
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
            if (group.flags & MeshFlags::IsOpaque) {
                // we group by flags, and thus can force the whole group to opaque.
                // TODO: Maybe reduce groups by only setting this in geometry level?
                geometry_instance_flags |= vk::GeometryInstanceFlagBitsKHR::eForceOpaque;
            }
            if (group.flags & MeshFlags::FrontCounterClockwise) {
                geometry_instance_flags |=
                    vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise;
            }
            if (group.flags & MeshFlags::TwoSided) {
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
    }

    // needs to be moved once nodes support animating the transforms
    if (true /*needs_regroup || transforms_changed*/) {
        // TODO: only run if transforms changed (or in previous transformations changed)
        upload_transforms(cmd);
    }

    // Allocate/upload vertex/index buffers first so their device addresses are available
    // when GeometryData is written below.
    upload_meshes(cmd);

    // changes at regroup, or updated mesh (buffer address) -> run every frame.
    upload_geometry_data(cmd);

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    });

    if (true /*build_as*/) {
        build_blas(cmd);
        // TODO only build if blases where built or instances where modified.
        // TODO: compact the static BLASes -> reduce memory bandwidth and increase performance.

        // TODO: only call if a BLAS was built / updated or transforms changed.
        build_tlas(cmd);

        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer |
                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eTransferWrite |
                vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
        });
    }

    needs_regroup = false;
    transforms_changed = false;

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
