#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/slang_program.hpp"
#include "merian/utils/normal_encoding.hpp"

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
      obj_allocator(obj_allocator), material_system(material_system) {

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
    ensure_index_vertex_buffers(INITIAL_INDEX_BUFFER_COUNT, INITIAL_VERTEX_BUFFER_COUNT);

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

void Scene::ensure_index_vertex_buffers(const std::size_t min_index_buffer_count,
                                        const std::size_t min_vertex_buffer_count) {
    if (min_index_buffer_count > index_buffers.size() ||
        min_vertex_buffer_count > vertex_buffers.size()) {

        index_buffers.resize(min_index_buffer_count);
        vertex_buffers.resize(min_index_buffer_count);
        prev_vertex_buffers.resize(min_index_buffer_count);

        composition->add_module_from_string(
            "scene_constants",
            fmt::format("namespace merian {{\n"
                        "export static const int merian_scene_index_buffers_count = {};\n"
                        "export static const int merian_scene_vertex_buffers_count = {};\n"
                        "export static const int merian_scene_prev_vertex_buffers_count = {};\n"
                        "}}",
                        static_cast<uint32_t>(index_buffers.size()),
                        static_cast<uint32_t>(vertex_buffers.size()),
                        static_cast<uint32_t>(prev_vertex_buffers.size())));

        rebuild_shader_object();
    }
}

void Scene::rebuild_shader_object() {
    if (!layout_program) {
        layout_program = SlangProgram::create(compile_context, composition);
    }

    shader_object = layout_program->create_shader_object(context, "merian::Scene", obj_allocator);
    shader_object->get_cursor()["material_system"] = material_system;
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
    auto id = static_cast<MeshID>(meshes.size());
    meshes.push_back(std::move(mesh));

    // note we ignore the mesh as long there are no instances
    // in add_mesh_instance the mesh is marked dirty and regrouping is enfored.

    return id;
}

NodeID Scene::add_node(SceneNode node) {
    auto id = static_cast<NodeID>(scene_graph.size());
    if (node.parent != NODE_ID_INVALID) {
        assert(node.parent < scene_graph.size());

        SceneNode& parent = scene_graph[node.parent];
        parent.children.push_back(id);
        assert(parent.global_transform);
        node.global_transform = mul(*parent.global_transform, node.local_transform);
    } else {
        node.global_transform = node.local_transform;
    }
    node.global_inverse_transposed = inverse(transpose(*node.global_transform));

    scene_graph.push_back(std::move(node));
    return id;
}

void Scene::add_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_id < meshes.size());
    assert(node_id < scene_graph.size());

    if (meshes[mesh_id]->instances.size() == 0 ||
        (!(meshes[mesh_id]->flags & GeometryFlags::IsDynamic) &&
         meshes[mesh_id]->instances.size() == 1)) {
        // - instances was 0 -> needs upload
        // - instances was 1 and is static: previously we could pretransform, but now this is not
        // possible anymore.
        mark_mesh_dirty(mesh_id);
    }
    meshes[mesh_id]->instances.insert(node_id);
    needs_regroup = true;
}

void Scene::mark_mesh_dirty(const MeshID mesh_id) {
    assert(mesh_id < meshes.size());
    assert((get_mesh(mesh_id).flags & GeometryFlags::IsDynamic) == 0);
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
    for (MeshID mesh_id = 0; mesh_id < meshes.size(); mesh_id++) {
        if (meshes[mesh_id]->flags & GeometryFlags::IsDynamic) {
            mark_mesh_dirty(mesh_id);
        }
    }
}

void Scene::node_properties(Properties& props, const SceneNode& node) {
    props.output_text("{}", node);

    for (uint32_t i = 0; i < node.children.size(); i++) {
        const SceneNode& child = scene_graph[node.children[i]];
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
            for (uint32_t id = 0; id < meshes.size(); id++) {
                if (props.st_begin_child(fmt::format("mesh_{:02} ", id),
                                         fmt::format("{:02}: {}", id, meshes[id]->name))) {
                    props.output_text("{}", *meshes[id]);
                    props.st_end_child();
                }
            }
            props.st_end_child();
        }

        if (props.st_begin_child("graph", "Graph")) {
            for (uint32_t id = 0; id < scene_graph.size(); id++) {
                const SceneNode& node = scene_graph[id];
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

    if (props.st_begin_child("stats", "Statistics")) {
        std::size_t total_vertices = 0;
        std::size_t total_triangles = 0;
        for (const auto& m : meshes) {
            total_vertices += m->get_vertex_count();
            total_triangles += m->get_primitive_count();
        }

        props.output_text("nodes: {}\nmeshes: {}\nvertices: {}\ntriangles: {}\nmaterials: "
                          "{}\ntextures: {}",
                          scene_graph.size(), meshes.size(), total_vertices, total_triangles,
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
    auto prev_mesh_groups_static_non_instanced = std::move(mesh_groups_static_non_instanced);
    auto prev_mesh_groups_dynamic_non_instanced = std::move(mesh_groups_dynamic_non_instanced);
    auto prev_mesh_groups_instanced = std::move(mesh_groups_instanced);

    mesh_groups.clear();
    prev_mesh_groups_static_non_instanced.clear();
    prev_mesh_groups_dynamic_non_instanced.clear();
    prev_mesh_groups_instanced.clear();

    // See description in scene.hpp for grouping logic

    for (MeshID mid = 0; mid < static_cast<MeshID>(meshes.size()); mid++) {
        const Mesh& mesh = *meshes[mid];
        if (mesh.instances.empty()) {
            continue;
        }

        if (mesh.instances.size() > 1) {
            // is instanced

        } else {
            if (mesh.flags & GeometryFlags::IsDynamic) {
                // dynamic, non-instanced
            } else {
                // static, non-instanced
            }
        }
    }

    // // Classify meshes into groups:
    // // 1. Non-instanced static -> split into opaque / non-opaque buckets so the
    // //    TLAS instance flags can carry the eForceOpaque bit per BLAS.
    // // 2. Non-instanced dynamic -> grouped by NodeID (same transform)
    // // 3. Instanced -> grouped by identical instance set
    // std::vector<MeshID> static_opaque;
    // std::vector<MeshID> static_alpha;
    // std::unordered_map<NodeID, std::vector<MeshID>> dynamic_by_node;
    // std::map<std::set<NodeID>, std::vector<MeshID>> instanced_by_set;

    // for (MeshID mid = 0; mid < static_cast<MeshID>(meshes.size()); mid++) {
    //     const Mesh& mesh = *meshes[mid];
    //     if (mesh.instances.empty())
    //         continue;

    //     if (mesh.instances.size() == 1) {
    //         const bool is_dynamic = mesh.flags & GeometryFlags::IsDynamic;
    //         if (is_dynamic && !pretransform_dynamic) {
    //             dynamic_by_node[*mesh.instances.begin()].push_back(mid);
    //         } else if (mesh.flags & GeometryFlags::IsOpaque) {
    //             static_opaque.push_back(mid);
    //         } else {
    //             static_alpha.push_back(mid);
    //         }
    //     } else {
    //         // Instanced: group by identical instance set
    //         instanced_by_set[mesh.instances].push_back(mid);
    //     }
    // }

    // // Build groups. is_opaque is set per group so build_tlas can apply
    // // eForceOpaque without re-walking the meshes.
    // auto group_is_opaque = [&](const std::vector<MeshID>& list) {
    //     for (MeshID mid : list) {
    //         if (!(meshes[mid]->flags & GeometryFlags::IsOpaque))
    //             return false;
    //     }
    //     return true;
    // };

    // if (!static_opaque.empty()) {
    //     mesh_groups.push_back({static_opaque, true, true});
    // }
    // if (!static_alpha.empty()) {
    //     mesh_groups.push_back({static_alpha, true, false});
    // }
    // for (auto& [node_id, mesh_list] : dynamic_by_node) {
    //     const bool opaque = group_is_opaque(mesh_list);
    //     mesh_groups.push_back({std::move(mesh_list), false, opaque});
    // }
    // for (auto& [instance_set, mesh_list] : instanced_by_set) {
    //     const bool opaque = group_is_opaque(mesh_list);
    //     mesh_groups.push_back({std::move(mesh_list), false, opaque});
    // }

    // // Build geometry instance data and per-mesh instance ID mapping.
    // // Order: for each group, for each instance of the group, geometries in mesh_list order.
    // // This ensures InstanceID() + GeometryIndex() directly indexes geometry_instance_data.
    // mesh_id_to_instance_ids.clear();
    // mesh_id_to_instance_ids.resize(meshes.size());
    // mesh_id_to_group_id.assign(meshes.size(), UINT32_MAX);
    // geometry_instance_data.clear();

    // uint32_t geometry_instance_id = 0;
    // for (uint32_t gi = 0; gi < mesh_groups.size(); gi++) {
    //     const auto& group = mesh_groups[gi];
    //     // All meshes in a group have the same instance set (for non-instanced: size 1)
    //     assert(!group.mesh_list.empty());
    //     const auto& instances = meshes[group.mesh_list[0]]->instances;
    //     uint32_t instance_count = static_cast<uint32_t>(instances.size());

    //     for (MeshID mid : group.mesh_list) {
    //         mesh_id_to_group_id[mid] = gi;
    //     }

    //     for (uint32_t inst_idx = 0; inst_idx < instance_count; inst_idx++) {
    //         for (uint32_t geom_idx = 0; geom_idx < group.mesh_list.size(); geom_idx++) {
    //             MeshID mid = group.mesh_list[geom_idx];
    //             mesh_id_to_instance_ids[mid].push_back(geometry_instance_id);

    //             GeometryData gd{};
    //             gd.material_id = meshes[mid]->material_id;
    //             gd.vertex_buffer_index = static_cast<uint16_t>(mid);
    //             gd.index_buffer_index = static_cast<uint16_t>(mid);
    //             gd.flags = group.is_static ? GEOMETRY_FLAG_PRETRANSFORMED : 0;
    //             geometry_instance_data.push_back(gd);

    //             geometry_instance_id++;
    //         }
    //     }
//}
}

// void Scene::upload_geometry_buffers(const CommandBufferHandle& cmd) {
//     if (!geometry_dirty)
//         return;

//     create_mesh_groups();

//     // Upload one vertex buffer and one index buffer per mesh
//     vertex_buffers.clear();
//     index_buffers.clear();
//     vertex_buffers.resize(meshes.size());
//     index_buffers.resize(meshes.size());

//     // Meshes uploaded in a static group are baked into world space on CPU.
//     std::vector<bool> pretransform_mesh(meshes.size(), false);
//     for (const auto& group : mesh_groups) {
//         if (!group.is_static)
//             continue;
//         for (MeshID mid : group.mesh_list)
//             pretransform_mesh[mid] = true;
//     }

//     const auto staging = allocator->get_staging();

//     for (MeshID mid = 0; mid < static_cast<MeshID>(meshes.size()); mid++) {
//         const Mesh& mesh = *meshes[mid];
//         if (mesh.instances.empty())
//             continue;

//         const uint32_t vertex_count = mesh.get_vertex_count();
//         const uint32_t primitive_count = mesh.get_primitive_count();
//         const vk::DeviceSize vb_size = vertex_count * sizeof(PackedVertexData);
//         const vk::DeviceSize ib_size = primitive_count * sizeof(uint3);

//         const SceneNode* pretransform_node = nullptr;
//         if (pretransform_mesh[mid]) {
//             pretransform_node = &scene_graph[*mesh.instances.begin()];
//         }

//         const auto buffer_usage =
//             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
//             vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
//             vk::BufferUsageFlagBits::eShaderDeviceAddress;

//         auto vb = allocator->create_buffer(vb_size, buffer_usage, MemoryMappingType::NONE,
//                                            fmt::format("Scene::vb[{}]", mid));
//         const MemoryAllocationHandle vb_staging = staging->cmd_to_device(cmd, vb);
//         auto* vb_mapped = static_cast<PackedVertexData*>(vb_staging->map());
//         if (pretransform_node) {
//             for (uint32_t v = 0; v < vertex_count; v++) {
//                 vb_mapped[v] = mesh.get_packed_vertex_pretransformed(v, *pretransform_node);
//             }
//         } else {
//             for (uint32_t v = 0; v < vertex_count; v++) {
//                 vb_mapped[v] = mesh.get_packed_vertex(v);
//             }
//         }
//         vb_staging->unmap();
//         vertex_buffers[mid] = vb;

//         auto ib = allocator->create_buffer(ib_size, buffer_usage, MemoryMappingType::NONE,
//                                            fmt::format("Scene::ib[{}]", mid));
//         const MemoryAllocationHandle ib_staging = staging->cmd_to_device(cmd, ib);
//         auto* ib_mapped = static_cast<uint3*>(ib_staging->map());
//         for (uint32_t p = 0; p < primitive_count; p++) {
//             ib_mapped[p] = mesh.get_indices(p);
//         }
//         ib_staging->unmap();
//         index_buffers[mid] = ib;
//     }

//     // Upload geometry instance data
//     if (!geometry_instance_data.empty()) {
//         geometry_data_buffer = allocator->create_buffer(
//             geometry_instance_data.size() * sizeof(GeometryData),
//             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
//             MemoryMappingType::NONE, "Scene::geometry_data");
//         allocator->get_staging()->cmd_to_device(
//             cmd, geometry_data_buffer, geometry_instance_data.data(), 0,
//             geometry_instance_data.size() * sizeof(GeometryData));

//         uint32_t tlas_instance_count = 0;
//         for (const auto& group : mesh_groups) {
//             tlas_instance_count +=
//                 static_cast<uint32_t>(meshes[group.mesh_list[0]]->instances.size());
//         }

//         const auto transform_size = tlas_instance_count * sizeof(float4x4);
//         const auto transform_usage =
//             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
//         instance_transforms_buffer = allocator->create_buffer(
//             transform_size, transform_usage, MemoryMappingType::NONE,
//             "Scene::instance_transforms");
//         inverse_transposed_instance_transforms_buffer =
//             allocator->create_buffer(transform_size, transform_usage, MemoryMappingType::NONE,
//                                      "Scene::inv_transposed_instance_transforms");
//         prev_instance_transforms_buffer =
//             allocator->create_buffer(transform_size, transform_usage, MemoryMappingType::NONE,
//                                      "Scene::prev_instance_transforms");
//         prev_inverse_transposed_instance_transforms_buffer =
//             allocator->create_buffer(transform_size, transform_usage, MemoryMappingType::NONE,
//                                      "Scene::prev_inv_transposed_instance_transforms");
//         prev_instance_transforms_data.clear();
//     }

//     // Barrier: all geometry transfers → shader reads + AS build reads
//     cmd->barrier(vk::MemoryBarrier2{
//         vk::PipelineStageFlagBits2::eTransfer,
//         vk::AccessFlagBits2::eTransferWrite,
//         vk::PipelineStageFlagBits2::eAllCommands,
//         vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
//     });

//     update_composition_constants();
//     rebuild_shader_object();
//     geometry_dirty = false;
// }

// void Scene::refresh_dirty_mesh_buffers(const CommandBufferHandle& cmd) {
//     if (mesh_data_dirty.empty())
//         return;

//     const auto buffer_usage = vk::BufferUsageFlagBits::eStorageBuffer |
//                               vk::BufferUsageFlagBits::eTransferDst |
//                               vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
//                               | vk::BufferUsageFlagBits::eShaderDeviceAddress;

//     const auto staging = allocator->get_staging();
//     bool any_uploaded = false;

//     for (MeshID mid = 0; mid < static_cast<MeshID>(mesh_data_dirty.size()); mid++) {
//         if (!mesh_data_dirty[mid])
//             continue;
//         assert(mid < meshes.size());
//         const Mesh& mesh = *meshes[mid];
//         if (mesh.instances.empty())
//             continue;

//         // Refresh-dirty meshes are dynamic by contract (caller guarantees they
//         // are not in a static / pretransformed group).
//         const uint32_t vertex_count = mesh.get_vertex_count();
//         const uint32_t primitive_count = mesh.get_primitive_count();
//         const vk::DeviceSize vb_size = vertex_count * sizeof(PackedVertexData);
//         const vk::DeviceSize ib_size = primitive_count * sizeof(uint3);

//         if (vb_size == 0 || ib_size == 0)
//             continue;

//         if (mid >= vertex_buffers.size())
//             vertex_buffers.resize(mid + 1);
//         if (mid >= index_buffers.size())
//             index_buffers.resize(mid + 1);

//         allocator->ensure_buffer_size(vertex_buffers[mid], vb_size, buffer_usage,
//                                       fmt::format("Scene::vb[{}]", mid), MemoryMappingType::NONE,
//                                       std::nullopt, 1.5f);
//         allocator->ensure_buffer_size(index_buffers[mid], ib_size, buffer_usage,
//                                       fmt::format("Scene::ib[{}]", mid), MemoryMappingType::NONE,
//                                       std::nullopt, 1.5f);

//         const MemoryAllocationHandle vb_staging = staging->cmd_to_device(cmd,
//         vertex_buffers[mid]); auto* vb_mapped =
//         static_cast<PackedVertexData*>(vb_staging->map()); for (uint32_t v = 0; v < vertex_count;
//         v++) {
//             vb_mapped[v] = mesh.get_packed_vertex(v);
//         }
//         vb_staging->unmap();

//         const MemoryAllocationHandle ib_staging = staging->cmd_to_device(cmd,
//         index_buffers[mid]); auto* ib_mapped = static_cast<uint3*>(ib_staging->map()); for
//         (uint32_t p = 0; p < primitive_count; p++) {
//             ib_mapped[p] = mesh.get_indices(p);
//         }
//         ib_staging->unmap();
//         any_uploaded = true;
//     }

//     if (any_uploaded) {
//         cmd->barrier(vk::MemoryBarrier2{
//             vk::PipelineStageFlagBits2::eTransfer,
//             vk::AccessFlagBits2::eTransferWrite,
//             vk::PipelineStageFlagBits2::eAllCommands,
//             vk::AccessFlagBits2::eShaderRead |
//             vk::AccessFlagBits2::eAccelerationStructureReadKHR,
//         });
//     }
// }

// void Scene::build_blas(const CommandBufferHandle& cmd, const std::vector<bool>& needs_build) {
//     if (!build_as || mesh_groups.empty())
//         return;

//     assert(needs_build.size() == mesh_groups.size());

//     if (!as_builder)
//         as_builder.emplace(context, allocator);

//     // Store geometry/range data so pointers remain valid until get_cmds
//     struct BLASGeom {
//         std::vector<vk::AccelerationStructureGeometryKHR> geometries;
//         std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
//     };
//     std::vector<BLASGeom> blas_geoms(mesh_groups.size());

//     for (uint32_t gi = 0; gi < mesh_groups.size(); gi++) {
//         if (!needs_build[gi])
//             continue;
//         const auto& group = mesh_groups[gi];
//         auto& bg = blas_geoms[gi];

//         for (MeshID mid : group.mesh_list) {
//             const Mesh& mesh = *meshes[mid];

//             vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
//             triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
//             triangles.vertexData = vertex_buffers[mid]->get_device_address();
//             triangles.vertexStride = sizeof(PackedVertexData);
//             triangles.maxVertex = mesh.get_vertex_count() - 1;
//             triangles.indexType = vk::IndexType::eUint32;
//             triangles.indexData = index_buffers[mid]->get_device_address();

//             vk::AccelerationStructureGeometryKHR geom;
//             geom.geometryType = vk::GeometryTypeKHR::eTriangles;
//             geom.geometry.triangles = triangles;
//             if (mesh.flags & GeometryFlags::IsOpaque) {
//                 geom.flags = vk::GeometryFlagBitsKHR::eOpaque;
//             }

//             vk::AccelerationStructureBuildRangeInfoKHR range{};
//             range.primitiveCount = mesh.get_primitive_count();

//             bg.geometries.push_back(geom);
//             bg.ranges.push_back(range);
//         }
//     }

//     // Preserve unchanged BLAS handles, queue new builds for the dirty groups.
//     // Build-time / trace-time tradeoff: static groups are built once, so prefer
//     // fast trace; dynamic groups are rebuilt every frame, so prefer fast build.
//     std::vector<AccelerationStructureHandle> new_blas_list(mesh_groups.size());
//     bool any_queued = false;
//     for (uint32_t gi = 0; gi < mesh_groups.size(); gi++) {
//         if (!needs_build[gi]) {
//             assert(gi < blas_list.size() && blas_list[gi] &&
//                    "needs_build[gi]=false requires an existing BLAS to preserve");
//             new_blas_list[gi] = blas_list[gi];
//             continue;
//         }
//         const vk::BuildAccelerationStructureFlagsKHR flags =
//             mesh_groups[gi].is_static ?
//             vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
//                                       :
//                                       vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
//         new_blas_list[gi] =
//             as_builder->queue_build(blas_geoms[gi].geometries, blas_geoms[gi].ranges, flags);
//         any_queued = true;
//     }

//     // Record commands and allocate scratch only if at least one build was queued.
//     if (any_queued) {
//         as_builder->get_cmds_blas(cmd, scratch_buffer);
//     }

//     blas_list = std::move(new_blas_list);
// }

// void Scene::build_tlas(const CommandBufferHandle& cmd) {
//     if (!build_as || blas_list.empty())
//         return;

//     std::vector<vk::AccelerationStructureInstanceKHR> instances;
//     uint32_t instance_id = 0;

//     for (uint32_t gi = 0; gi < mesh_groups.size(); gi++) {
//         const auto& group = mesh_groups[gi];
//         const auto& first_mesh_instances = meshes[group.mesh_list[0]]->instances;
//         uint32_t instance_count = static_cast<uint32_t>(first_mesh_instances.size());

//         // Iterate each instance of this group
//         auto node_it = first_mesh_instances.begin();
//         for (uint32_t inst_idx = 0; inst_idx < instance_count; inst_idx++, ++node_it) {
//             vk::AccelerationStructureInstanceKHR inst{};

//             if (group.is_static) {
//                 // Static: identity transform (meshes pre-transformed)
//                 inst.transform.matrix[0][0] = 1.f;
//                 inst.transform.matrix[1][1] = 1.f;
//                 inst.transform.matrix[2][2] = 1.f;
//             } else {
//                 // Dynamic: apply node's global transform
//                 NodeID nid = *node_it;
//                 const auto& t = *scene_graph[nid].global_transform;
//                 for (int row = 0; row < 3; row++)
//                     for (int col = 0; col < 4; col++)
//                         inst.transform.matrix[row][col] = t[row][col];
//             }

//             inst.instanceCustomIndex = instance_id;
//             inst.mask = 0xFF;
//             inst.accelerationStructureReference =
//                 blas_list[gi]->get_acceleration_structure_device_address();

//             vk::GeometryInstanceFlagsKHR inst_flags{};
//             if (group.is_opaque) {
//                 inst_flags |= vk::GeometryInstanceFlagBitsKHR::eForceOpaque;
//             }
//             // Per-mesh winding flag: in this BLAS layout one mesh group is one
//             // BLAS, so all meshes in a group share an instance flag set. The
//             // builder picks the first mesh's winding; mixing is not supported.
//             const auto& head_flags = meshes[group.mesh_list[0]]->flags;
//             if (head_flags & GeometryFlags::FrontCounterClockwise) {
//                 inst_flags |= vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise;
//             }
//             if (head_flags & GeometryFlags::TwoSided) {
//                 inst_flags |= vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable;
//             }
//             inst.flags = static_cast<uint32_t>(inst_flags);

//             instances.push_back(inst);
//             instance_id += static_cast<uint32_t>(group.mesh_list.size());
//         }
//     }

//     if (instances.empty())
//         return;

//     const vk::DeviceSize required_size =
//         instances.size() * sizeof(vk::AccelerationStructureInstanceKHR);
//     if (!tlas_instances_buffer || tlas_instances_buffer->get_size() < required_size) {
//         // Buffer must grow (or first allocation). Park the previous buffer in the
//         // keepalive ring so any in-flight command buffer that still references
//         // it stays valid; the slot we overwrite was last touched
//         // TLAS_INSTANCES_KEEPALIVE frames ago.
//         tlas_instances_keepalive[tlas_instances_keepalive_idx] = tlas_instances_buffer;
//         tlas_instances_keepalive_idx =
//             (tlas_instances_keepalive_idx + 1) % TLAS_INSTANCES_KEEPALIVE;
//         tlas_instances_buffer = allocator->create_buffer(
//             required_size,
//             vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
//                 vk::BufferUsageFlagBits::eTransferDst |
//                 vk::BufferUsageFlagBits::eShaderDeviceAddress,
//             MemoryMappingType::NONE, "Scene::tlas_instances");
//     }
//     allocator->get_staging()->cmd_to_device(cmd, tlas_instances_buffer, instances.data(), 0,
//                                             required_size);

//     cmd->barrier(tlas_instances_buffer->buffer_barrier2(
//         vk::PipelineStageFlagBits2::eTransfer,
//         vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
//         vk::AccessFlagBits2::eTransferWrite,
//         vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead));

//     tlas = as_builder->queue_build(static_cast<uint32_t>(instances.size()),
//     tlas_instances_buffer); as_builder->get_cmds_tlas(cmd, scratch_buffer);
// }

void Scene::update(const CommandBufferHandle& cmd,
                   const float time,
                   const float time_diff,
                   const uint32_t frame) {
    on_update(cmd, time, time_diff, frame);

    assert(!cameras.empty() &&
           "the scene implementation must ensure that there is at least one camera");

    material_system->update(cmd);

    if (needs_regroup) {
        compute_mesh_groups();
    }

    // // upload_geometry_buffers re-uploads everything; in that case any per-mesh
    // // dirty bits are subsumed and cleared.
    // const bool full_geometry_rebuild = geometry_dirty;
    // upload_geometry_buffers(cmd);
    // if (full_geometry_rebuild) {
    //     std::fill(mesh_data_dirty.begin(), mesh_data_dirty.end(), false);
    // }

    // // Otherwise, refresh just the dirty meshes' vertex/index data and figure
    // // out which BLAS groups must be rebuilt.
    // std::vector<bool> group_needs_build;
    // bool any_group_dirty = false;
    // if (!full_geometry_rebuild && !mesh_data_dirty.empty()) {
    //     refresh_dirty_mesh_buffers(cmd);
    //     group_needs_build.assign(mesh_groups.size(), false);
    //     for (MeshID mid = 0; mid < static_cast<MeshID>(mesh_data_dirty.size()); mid++) {
    //         if (!mesh_data_dirty[mid])
    //             continue;
    //         if (mid >= mesh_id_to_group_id.size())
    //             continue;
    //         const uint32_t gi = mesh_id_to_group_id[mid];
    //         if (gi == UINT32_MAX)
    //             continue;
    //         group_needs_build[gi] = true;
    //         any_group_dirty = true;
    //     }
    //     std::fill(mesh_data_dirty.begin(), mesh_data_dirty.end(), false);
    // }

    // if (build_as && (bvh_dirty || any_group_dirty)) {
    //     std::vector<bool> needs_build(mesh_groups.size(), false);
    //     if (bvh_dirty) {
    //         std::fill(needs_build.begin(), needs_build.end(), true);
    //     } else {
    //         needs_build = std::move(group_needs_build);
    //     }
    //     build_blas(cmd, needs_build);
    //     build_tlas(cmd);
    //     bvh_dirty = false;
    // }

    // // Upload instance transforms each frame (dynamic groups change every frame).
    // // One transform per TLAS instance, ordered to match build_tlas() so InstanceIndex() indexes
    // it. if (instance_transforms_buffer && !geometry_instance_data.empty()) {
    //     uint32_t tlas_instance_count = 0;
    //     for (const auto& group : mesh_groups) {
    //         tlas_instance_count +=
    //             static_cast<uint32_t>(meshes[group.mesh_list[0]]->instances.size());
    //     }
    //     std::vector<float4x4> transforms(tlas_instance_count, identity());
    //     uint32_t tlas_inst_id = 0;
    //     for (const auto& group : mesh_groups) {
    //         const auto& instances = meshes[group.mesh_list[0]]->instances;
    //         auto node_it = instances.begin();
    //         for (uint32_t inst_idx = 0; inst_idx < instances.size(); inst_idx++, ++node_it) {
    //             if (!group.is_static) {
    //                 transforms[tlas_inst_id] = *scene_graph[*node_it].global_transform;
    //             }
    //             tlas_inst_id++;
    //         }
    //     }

    //     // Use current transforms as prev on first frame or after geometry rebuild.
    //     if (prev_instance_transforms_data.size() != transforms.size())
    //         prev_instance_transforms_data = transforms;

    //     // Compute inverse-transposed transforms for normal/direction transformation.
    //     std::vector<float4x4> inv_transposed(transforms.size());
    //     std::vector<float4x4> prev_inv_transposed(transforms.size());
    //     for (size_t i = 0; i < transforms.size(); i++) {
    //         inv_transposed[i] = inverse(transpose(transforms[i]));
    //         prev_inv_transposed[i] = inverse(transpose(prev_instance_transforms_data[i]));
    //     }

    //     const auto buf_size = transforms.size() * sizeof(float4x4);
    //     auto staging = allocator->get_staging();
    //     staging->cmd_to_device(cmd, instance_transforms_buffer, transforms.data(), 0, buf_size);
    //     staging->cmd_to_device(cmd, inverse_transposed_instance_transforms_buffer,
    //                            inv_transposed.data(), 0, buf_size);
    //     staging->cmd_to_device(cmd, prev_instance_transforms_buffer,
    //                            prev_instance_transforms_data.data(), 0, buf_size);
    //     staging->cmd_to_device(cmd, prev_inverse_transposed_instance_transforms_buffer,
    //                            prev_inv_transposed.data(), 0, buf_size);

    //     cmd->barrier(vk::MemoryBarrier2{
    //         vk::PipelineStageFlagBits2::eTransfer,
    //         vk::AccessFlagBits2::eTransferWrite,
    //         vk::PipelineStageFlagBits2::eAllCommands,
    //         vk::AccessFlagBits2::eShaderRead,
    //     });

    //     // Save current transforms as previous for next frame.
    //     prev_instance_transforms_data = std::move(transforms);
    // }

    // // Lazy placeholder buffer: keeps every Scene descriptor binding bound
    // // even before any mesh has been added (consumers like gbuffer_rt gate
    // // on has_geometry() before dispatching but still need valid descriptor
    // // writes). See the placeholder_buffer comment in scene.hpp.
    // // TODO: skip this when VK_EXT_robustness2 nullDescriptor is enabled.
    // if (!placeholder_buffer) {
    //     const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
    //                        vk::BufferUsageFlagBits::eTransferDst |
    //                        vk::BufferUsageFlagBits::eShaderDeviceAddress |
    //                        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    //     placeholder_buffer = allocator->create_buffer(
    //         sizeof(float4x4), usage, MemoryMappingType::NONE, "Scene::placeholder");
    // }

    auto c = shader_object->get_cursor();

    // The slang composition declares the arrays with size max(N, 1), so
    // slot 0 always exists and must be bound. Fall back to placeholder
    // when the real mesh buffer isn't available.
    {
        const BufferHandle& vb0 =
            (!vertex_buffers.empty() && vertex_buffers[0]) ? vertex_buffers[0] : placeholder_buffer;
        const BufferHandle& ib0 =
            (!index_buffers.empty() && index_buffers[0]) ? index_buffers[0] : placeholder_buffer;
        c["vertex_buffers"][0] = vb0;
        c["index_buffers"][0] = ib0;
    }
    for (uint32_t i = 1; i < vertex_buffers.size(); i++) {
        if (vertex_buffers[i])
            c["vertex_buffers"][i] = vertex_buffers[i];
    }
    for (uint32_t i = 1; i < index_buffers.size(); i++) {
        if (index_buffers[i])
            c["index_buffers"][i] = index_buffers[i];
    }

    c["geometries"] = geometry_data_buffer ? geometry_data_buffer : placeholder_buffer;
    c["instance_transforms"] =
        instance_transforms_buffer ? instance_transforms_buffer : placeholder_buffer;
    c["inverse_transposed_instance_transforms"] =
        inverse_transposed_instance_transforms_buffer
            ? inverse_transposed_instance_transforms_buffer
            : placeholder_buffer;
    c["prev_instance_transforms"] =
        prev_instance_transforms_buffer ? prev_instance_transforms_buffer : placeholder_buffer;
    c["prev_inverse_transposed_instance_transforms"] =
        prev_inverse_transposed_instance_transforms_buffer
            ? prev_inverse_transposed_instance_transforms_buffer
            : placeholder_buffer;

    c["material_system"] = material_system->get_shader_object();
    c["frame"] = frame;
    c["time"] = get_time(time);
    c["time_diff"] = time_diff;

    if (build_as) {
        AccelerationStructureHandle bind_tlas = tlas;
        if (!bind_tlas) {
            // Build a one-shot empty TLAS (0 instances) so the AS binding
            // is always populated. Reuses placeholder_buffer for the
            // (unused) instance device address.
            if (!placeholder_tlas) {
                if (!as_builder)
                    as_builder.emplace(context, allocator);
                vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
                    VK_FALSE, {placeholder_buffer->get_device_address()}};
                placeholder_tlas = as_builder->queue_build(0u, instances_data);
                as_builder->get_cmds_tlas(cmd, scratch_buffer);
                cmd->barrier(vk::MemoryBarrier2{
                    vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                    vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                    vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eAccelerationStructureReadKHR,
                });
            }
            bind_tlas = placeholder_tlas;
        }
        c["as"]["as"] = bind_tlas;
    }

    auto cam = get_active_camera();
    assert(cam);

    prev_active_camera.write_to(c["prev_camera"]);
    cam->write_to(c["camera"]);
    prev_active_camera = *cam;
}

} // namespace merian
