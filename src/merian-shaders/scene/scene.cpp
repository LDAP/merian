#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/slang_program.hpp"
#include "merian/utils/normal_encoding.hpp"

#include <cassert>
#include <fmt/format.h>
#include <map>
#include <unordered_map>

namespace merian {

namespace {

Mesh apply_world_transform(const Mesh& mesh, const float4x4& M) {
    // this is expensive...
    Mesh out = mesh;

    float4x4 inverse_transposed = inverse(transpose(M));

    for (auto& vd : out.vertices) {
        vd.position = mul(M, float4(vd.position, 1));
        vd.encoded_normal = encode_normal(
            normalize(mul(inverse_transposed, float4(decode_normal(vd.encoded_normal), 0))));

        const uint32_t sign_bit = vd.encoded_tangent & 1u;
        const float3 t_obj = decode_normal(vd.encoded_tangent & ~1u);
        vd.encoded_tangent = (encode_normal(normalize(mul(M, float4(t_obj, 0)))) & ~1u) | sign_bit;
    }

    return out;
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
      obj_allocator(obj_allocator), material_system(material_system) {

    build_as = context->get_device()
                   ->get_enabled_features()
                   .get_acceleration_structure_features_khr()
                   .accelerationStructure == VK_TRUE;

    // Build composition once — subsequent changes modify in-place.
    composition = SlangComposition::create();
    composition->add_composition(material_system->get_composition());
    composition->add_module_from_path("merian-shaders/scene/scene.slang");
    composition->add_module_from_path("merian-shaders/scene/camera.slang");
    composition->add_module_from_path("merian-shaders/scene/environment-map.slang");
    composition->add_module_from_path("merian-shaders/scene/acceleration-structure.slang");
    update_composition_constants();

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

    layout_program = SlangProgram::create(compile_context, composition);
    rebuild_shader_object();
}

void Scene::update_composition_constants() {
    composition->add_module_from_string(
        "scene_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_scene_index_buffers_count = {};\n"
                    "export static const int merian_scene_vertex_buffers_count = {};\n"
                    "export static const int merian_scene_prev_vertex_buffers_count = {};\n"
                    "export static const int merian_scene_geometry_count = {};\n"
                    "}}",
                    static_cast<uint32_t>(std::max(index_buffers.size(), size_t(1))),
                    static_cast<uint32_t>(std::max(vertex_buffers.size(), size_t(1))), 0u,
                    static_cast<uint32_t>(std::max(geometry_instance_data.size(), size_t(1)))));
}

void Scene::rebuild_shader_object() {
    shader_object = layout_program->create_shader_object(context, "merian::Scene", obj_allocator);
    shader_object->get_cursor()["material_system"] = material_system;
}

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------

bool Scene::set_build_acceleration_structure(bool build) {
    if (build_as == build)
        return false;
    build_as = build;
    // TODO: rebuild composition once link-time AS type is re-enabled
    return true;
}

MeshID Scene::add_mesh(Mesh mesh) {
    auto id = static_cast<MeshID>(meshes.size());
    meshes.push_back(std::move(mesh));
    geometry_dirty = true;
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

    scene_graph.push_back(std::move(node));
    return id;
}

void Scene::add_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_id < meshes.size());
    assert(node_id < scene_graph.size());
    meshes[mesh_id].instances.insert(node_id);
    geometry_dirty = true;
    bvh_dirty = true;
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
    geometry_dirty = true;
    bvh_dirty = true;
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
                props.config_uint("active", active_camera, 0, cameras.size());
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
                                                       60.f, 1920.f / 1080.f, 0.01f, 1000.f);
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
                                         fmt::format("{:02}: {}", id, meshes[id].name))) {
                    props.output_text("{}", meshes[id]);
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
            total_vertices += m.vertices.size();
            total_triangles += m.indices.size();
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

void Scene::create_mesh_groups() {
    mesh_groups.clear();

    // Classify meshes into groups:
    // 1. Non-instanced static -> all in one group
    // 2. Non-instanced dynamic -> grouped by NodeID (same transform)
    // 3. Instanced -> grouped by identical instance set
    std::vector<MeshID> static_meshes;
    std::unordered_map<NodeID, std::vector<MeshID>> dynamic_by_node;
    std::map<std::set<NodeID>, std::vector<MeshID>> instanced_by_set;

    for (MeshID mid = 0; mid < static_cast<MeshID>(meshes.size()); mid++) {
        const auto& mesh = meshes[mid];
        if (mesh.instances.empty())
            continue;

        if (mesh.instances.size() == 1) {
            const bool is_dynamic = mesh.flags & GeometryFlags::IsDynamic;
            if (is_dynamic && !pretransform_dynamic) {
                dynamic_by_node[*mesh.instances.begin()].push_back(mid);
            } else {
                static_meshes.push_back(mid);
            }
        } else {
            // Instanced: group by identical instance set
            instanced_by_set[mesh.instances].push_back(mid);
        }
    }

    // Build groups
    if (!static_meshes.empty()) {
        mesh_groups.push_back({static_meshes, true});
    }
    for (auto& [node_id, mesh_list] : dynamic_by_node) {
        mesh_groups.push_back({std::move(mesh_list), false});
    }
    for (auto& [instance_set, mesh_list] : instanced_by_set) {
        mesh_groups.push_back({std::move(mesh_list), false});
    }

    // Build geometry instance data and per-mesh instance ID mapping.
    // Order: for each group, for each instance of the group, geometries in mesh_list order.
    // This ensures InstanceID() + GeometryIndex() directly indexes geometry_instance_data.
    mesh_id_to_instance_ids.clear();
    mesh_id_to_instance_ids.resize(meshes.size());
    geometry_instance_data.clear();

    uint32_t geometry_instance_id = 0;
    for (const auto& group : mesh_groups) {
        // All meshes in a group have the same instance set (for non-instanced: size 1)
        assert(!group.mesh_list.empty());
        const auto& instances = meshes[group.mesh_list[0]].instances;
        uint32_t instance_count = static_cast<uint32_t>(instances.size());

        for (uint32_t inst_idx = 0; inst_idx < instance_count; inst_idx++) {
            for (uint32_t geom_idx = 0; geom_idx < group.mesh_list.size(); geom_idx++) {
                MeshID mid = group.mesh_list[geom_idx];
                mesh_id_to_instance_ids[mid].push_back(geometry_instance_id);

                GeometryData gd{};
                gd.material_id = meshes[mid].material_id;
                gd.vertex_buffer_index = static_cast<uint16_t>(mid);
                gd.index_buffer_index = static_cast<uint16_t>(mid);
                gd.flags = group.is_static ? GEOMETRY_FLAG_PRETRANSFORMED : 0;
                geometry_instance_data.push_back(gd);

                geometry_instance_id++;
            }
        }
    }
}

void Scene::upload_geometry_buffers(const CommandBufferHandle& cmd) {
    if (!geometry_dirty)
        return;

    create_mesh_groups();

    // Upload one vertex buffer and one index buffer per mesh
    vertex_buffers.clear();
    index_buffers.clear();
    vertex_buffers.resize(meshes.size());
    index_buffers.resize(meshes.size());

    // Meshes uploaded in a static group are baked into world space on CPU.
    std::vector<bool> pretransform_mesh(meshes.size(), false);
    for (const auto& group : mesh_groups) {
        if (!group.is_static)
            continue;
        for (MeshID mid : group.mesh_list)
            pretransform_mesh[mid] = true;
    }

    for (MeshID mid = 0; mid < static_cast<MeshID>(meshes.size()); mid++) {
        const auto& mesh = meshes[mid];
        if (mesh.instances.empty())
            continue;

        const PackedVertexData* vertex_src = mesh.vertices.data();
        std::vector<PackedVertexData> baked_vertices;
        if (pretransform_mesh[mid]) {
            const NodeID nid = *mesh.instances.begin();
            const Mesh baked = apply_world_transform(mesh, *scene_graph[nid].global_transform);
            baked_vertices = std::move(baked.vertices);
            vertex_src = baked_vertices.data();
        }

        auto vb = allocator->create_buffer(
            mesh.vertices.size() * sizeof(PackedVertexData),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            MemoryMappingType::NONE, fmt::format("Scene::vb[{}]", mid));
        allocator->get_staging()->cmd_to_device(cmd, vb, vertex_src, 0,
                                                mesh.vertices.size() * sizeof(PackedVertexData));
        vertex_buffers[mid] = vb;

        auto ib = allocator->create_buffer(
            mesh.indices.size() * sizeof(uint3),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            MemoryMappingType::NONE, fmt::format("Scene::ib[{}]", mid));
        allocator->get_staging()->cmd_to_device(cmd, ib, mesh.indices.data(), 0,
                                                mesh.indices.size() * sizeof(uint3));
        index_buffers[mid] = ib;
    }

    // Upload geometry instance data
    if (!geometry_instance_data.empty()) {
        geometry_data_buffer = allocator->create_buffer(
            geometry_instance_data.size() * sizeof(GeometryData),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::NONE, "Scene::geometry_data");
        allocator->get_staging()->cmd_to_device(
            cmd, geometry_data_buffer, geometry_instance_data.data(), 0,
            geometry_instance_data.size() * sizeof(GeometryData));

        uint32_t tlas_instance_count = 0;
        for (const auto& group : mesh_groups) {
            tlas_instance_count += static_cast<uint32_t>(meshes[group.mesh_list[0]].instances.size());
        }

        const auto transform_size = tlas_instance_count * sizeof(float4x4);
        const auto transform_usage =
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
        instance_transforms_buffer = allocator->create_buffer(
            transform_size, transform_usage, MemoryMappingType::NONE, "Scene::instance_transforms");
        inverse_transposed_instance_transforms_buffer =
            allocator->create_buffer(transform_size, transform_usage, MemoryMappingType::NONE,
                                     "Scene::inv_transposed_instance_transforms");
        prev_instance_transforms_buffer =
            allocator->create_buffer(transform_size, transform_usage, MemoryMappingType::NONE,
                                     "Scene::prev_instance_transforms");
        prev_inverse_transposed_instance_transforms_buffer =
            allocator->create_buffer(transform_size, transform_usage, MemoryMappingType::NONE,
                                     "Scene::prev_inv_transposed_instance_transforms");
        prev_instance_transforms_data.clear();
    }

    // Barrier: all geometry transfers → shader reads + AS build reads
    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    });

    update_composition_constants();
    rebuild_shader_object();
    geometry_dirty = false;
}

void Scene::build_blas(const CommandBufferHandle& cmd) {
    if (!build_as || mesh_groups.empty())
        return;

    if (!as_builder)
        as_builder.emplace(context, allocator);

    // Store geometry/range data so pointers remain valid until get_cmds
    struct BLASGeom {
        std::vector<vk::AccelerationStructureGeometryKHR> geometries;
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
    };
    std::vector<BLASGeom> blas_geoms(mesh_groups.size());

    for (uint32_t gi = 0; gi < mesh_groups.size(); gi++) {
        const auto& group = mesh_groups[gi];
        auto& bg = blas_geoms[gi];

        for (MeshID mid : group.mesh_list) {
            const auto& mesh = meshes[mid];

            vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
            triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
            triangles.vertexData = vertex_buffers[mid]->get_device_address();
            triangles.vertexStride = sizeof(PackedVertexData);
            triangles.maxVertex = static_cast<uint32_t>(mesh.vertices.size() - 1);
            triangles.indexType = vk::IndexType::eUint32;
            triangles.indexData = index_buffers[mid]->get_device_address();

            vk::AccelerationStructureGeometryKHR geom;
            geom.geometryType = vk::GeometryTypeKHR::eTriangles;
            geom.geometry.triangles = triangles;
            if (mesh.flags & GeometryFlags::IsOpaque) {
                geom.flags = vk::GeometryFlagBitsKHR::eOpaque;
            }

            vk::AccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = static_cast<uint32_t>(mesh.indices.size());

            bg.geometries.push_back(geom);
            bg.ranges.push_back(range);
        }
    }

    // Queue all BLAS builds
    std::vector<AccelerationStructureHandle> new_blas_list;
    for (uint32_t gi = 0; gi < mesh_groups.size(); gi++) {
        auto& bg = blas_geoms[gi];
        auto blas = as_builder->queue_build(bg.geometries, bg.ranges);
        new_blas_list.push_back(blas);
    }

    // Build BLAS (this records commands and allocates scratch)
    as_builder->get_cmds_blas(cmd, scratch_buffer);

    // Keep the new BLAS list (previous ones released when shared_ptrs drop)
    blas_list = std::move(new_blas_list);
}

void Scene::build_tlas(const CommandBufferHandle& cmd) {
    if (!build_as || blas_list.empty())
        return;

    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    uint32_t instance_id = 0;

    for (uint32_t gi = 0; gi < mesh_groups.size(); gi++) {
        const auto& group = mesh_groups[gi];
        const auto& first_mesh_instances = meshes[group.mesh_list[0]].instances;
        uint32_t instance_count = static_cast<uint32_t>(first_mesh_instances.size());

        // Iterate each instance of this group
        auto node_it = first_mesh_instances.begin();
        for (uint32_t inst_idx = 0; inst_idx < instance_count; inst_idx++, ++node_it) {
            vk::AccelerationStructureInstanceKHR inst{};

            if (group.is_static) {
                // Static: identity transform (meshes pre-transformed)
                inst.transform.matrix[0][0] = 1.f;
                inst.transform.matrix[1][1] = 1.f;
                inst.transform.matrix[2][2] = 1.f;
            } else {
                // Dynamic: apply node's global transform
                NodeID nid = *node_it;
                const auto& t = *scene_graph[nid].global_transform;
                for (int row = 0; row < 3; row++)
                    for (int col = 0; col < 4; col++)
                        inst.transform.matrix[row][col] = t[row][col];
            }

            inst.instanceCustomIndex = instance_id;
            inst.mask = 0xFF;
            inst.accelerationStructureReference =
                blas_list[gi]->get_acceleration_structure_device_address();

            // Set opaque flag if all meshes in group are opaque
            bool all_opaque = true;
            for (MeshID mid : group.mesh_list) {
                if (!(meshes[mid].flags & GeometryFlags::IsOpaque)) {
                    all_opaque = false;
                    break;
                }
            }
            if (all_opaque) {
                inst.flags = static_cast<uint32_t>(vk::GeometryInstanceFlagBitsKHR::eForceOpaque);
            }

            instances.push_back(inst);
            instance_id += static_cast<uint32_t>(group.mesh_list.size());
        }
    }

    if (instances.empty())
        return;

    tlas_instances_buffer = allocator->create_buffer(
        instances.size() * sizeof(vk::AccelerationStructureInstanceKHR),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        MemoryMappingType::NONE, "Scene::tlas_instances");
    allocator->get_staging()->cmd_to_device(cmd, tlas_instances_buffer, instances.data(), 0,
                                            instances.size() *
                                                sizeof(vk::AccelerationStructureInstanceKHR));

    cmd->barrier(tlas_instances_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead));

    tlas = as_builder->queue_build(static_cast<uint32_t>(instances.size()), tlas_instances_buffer);
    as_builder->get_cmds_tlas(cmd, scratch_buffer);
}

void Scene::update(const CommandBufferHandle& cmd,
                   const float time,
                   const float time_diff,
                   const uint32_t frame) {
    on_update(time, time_diff, frame);

    assert(!cameras.empty() &&
           "the scene implementation must ensure that there is at least one camera");

    material_system->upload(cmd);
    upload_geometry_buffers(cmd);

    if (build_as && bvh_dirty) {
        build_blas(cmd);
        build_tlas(cmd);
        bvh_dirty = false;
    }

    // Upload instance transforms each frame (dynamic groups change every frame).
    // One transform per TLAS instance, ordered to match build_tlas() so InstanceIndex() indexes it.
    if (instance_transforms_buffer && !geometry_instance_data.empty()) {
        uint32_t tlas_instance_count = 0;
        for (const auto& group : mesh_groups) {
            tlas_instance_count +=
                static_cast<uint32_t>(meshes[group.mesh_list[0]].instances.size());
        }
        std::vector<float4x4> transforms(tlas_instance_count, identity());
        uint32_t tlas_inst_id = 0;
        for (const auto& group : mesh_groups) {
            const auto& instances = meshes[group.mesh_list[0]].instances;
            auto node_it = instances.begin();
            for (uint32_t inst_idx = 0; inst_idx < instances.size(); inst_idx++, ++node_it) {
                if (!group.is_static) {
                    transforms[tlas_inst_id] = *scene_graph[*node_it].global_transform;
                }
                tlas_inst_id++;
            }
        }

        // Use current transforms as prev on first frame or after geometry rebuild.
        if (prev_instance_transforms_data.size() != transforms.size())
            prev_instance_transforms_data = transforms;

        // Compute inverse-transposed transforms for normal/direction transformation.
        std::vector<float4x4> inv_transposed(transforms.size());
        std::vector<float4x4> prev_inv_transposed(transforms.size());
        for (size_t i = 0; i < transforms.size(); i++) {
            inv_transposed[i] = inverse(transpose(transforms[i]));
            prev_inv_transposed[i] = inverse(transpose(prev_instance_transforms_data[i]));
        }

        const auto buf_size = transforms.size() * sizeof(float4x4);
        auto staging = allocator->get_staging();
        staging->cmd_to_device(cmd, instance_transforms_buffer, transforms.data(), 0, buf_size);
        staging->cmd_to_device(cmd, inverse_transposed_instance_transforms_buffer,
                               inv_transposed.data(), 0, buf_size);
        staging->cmd_to_device(cmd, prev_instance_transforms_buffer,
                               prev_instance_transforms_data.data(), 0, buf_size);
        staging->cmd_to_device(cmd, prev_inverse_transposed_instance_transforms_buffer,
                               prev_inv_transposed.data(), 0, buf_size);

        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead,
        });

        // Save current transforms as previous for next frame.
        prev_instance_transforms_data = std::move(transforms);
    }

    auto c = shader_object->get_cursor();
    for (uint32_t i = 0; i < vertex_buffers.size(); i++) {
        if (vertex_buffers[i])
            c["vertex_buffers"][i] = vertex_buffers[i];
    }
    for (uint32_t i = 0; i < index_buffers.size(); i++) {
        if (index_buffers[i])
            c["index_buffers"][i] = index_buffers[i];
    }

    c["geometries"] = geometry_data_buffer;
    c["instance_transforms"] = instance_transforms_buffer;
    c["inverse_transposed_instance_transforms"] = inverse_transposed_instance_transforms_buffer;
    c["prev_instance_transforms"] = prev_instance_transforms_buffer;
    c["prev_inverse_transposed_instance_transforms"] =
        prev_inverse_transposed_instance_transforms_buffer;

    c["material_system"] = material_system->get_shader_object();
    c["frame"] = frame;
    c["time"] = time;
    c["time_diff"] = time_diff;

    if (build_as && tlas) {
        c["as"]["as"] = tlas;
    }

    auto cam = get_active_camera();
    if (cam) {
        prev_active_camera.write_to(c["prev_camera"]);
        cam->write_to(c["camera"]);
        prev_active_camera = *cam;
    }
}

} // namespace merian
