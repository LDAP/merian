#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/slang_program.hpp"

#include <cassert>
#include <fmt/format.h>
#include <map>
#include <unordered_map>

namespace merian {

// ---------------------------------------------------------------------------
// Composition & Layout
// ---------------------------------------------------------------------------

static SlangCompositionHandle
build_composition(const MaterialSystemHandle& material_system,
                  uint32_t index_buffer_count,
                  uint32_t vertex_buffer_count,
                  uint32_t prev_vertex_buffer_count,
                  uint32_t geometry_count,
                  bool build_as) {
    auto c = SlangComposition::create();
    c->add_composition(material_system->get_composition());
    c->add_module_from_path("merian-shaders/scene/scene.slang");
    c->add_module_from_path("merian-shaders/scene/camera.slang");
    c->add_module_from_path("merian-shaders/scene/environment-map.slang");
    c->add_module_from_path("merian-shaders/scene/acceleration-structure.slang");

    c->add_module_from_string(
        "scene_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_scene_index_buffers_count = {};\n"
                    "export static const int merian_scene_vertex_buffers_count = {};\n"
                    "export static const int merian_scene_prev_vertex_buffers_count = {};\n"
                    "export static const int merian_scene_geometry_count = {};\n"
                    "}}",
                    std::max(index_buffer_count, 1u), std::max(vertex_buffer_count, 1u),
                    std::max(prev_vertex_buffer_count, 1u), std::max(geometry_count, 1u)));

    // TODO: use link-time type for AS once Slang's lookupExternDeclRefType is fixed
    // if (build_as) {
    //     c->add_module_from_string(
    //         "scene_as_type",
    //         "import merian_shaders.scene.acceleration_structure;\n"
    //         "namespace merian { export struct SceneAccelerationStructure : AccelerationStructure "
    //         "= HWAccelerationStructure; }");
    // } else {
    //     c->add_module_from_string(
    //         "scene_as_type",
    //         "import merian_shaders.scene.acceleration_structure;\n"
    //         "namespace merian { export struct SceneAccelerationStructure : AccelerationStructure "
    //         "= NullAccelerationStructure; }");
    // }

    return c;
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

    build_as = context->get_device()
                   ->get_enabled_features()
                   .get_acceleration_structure_features_khr()
                   .accelerationStructure == VK_TRUE;

    rebuild_composition();
    rebuild_shader_object();
}

void Scene::rebuild_shader_object() {
    layout_program = SlangProgram::create(compile_context, composition);
    shader_object = layout_program->create_shader_object(context, "merian::Scene", obj_allocator);
    shader_object->get_cursor()["material_system"] = material_system;
}

void Scene::rebuild_composition() {
    composition = build_composition(
        material_system, static_cast<uint32_t>(std::max(index_buffers.size(), size_t(1))),
        static_cast<uint32_t>(std::max(vertex_buffers.size(), size_t(1))), 0,
        static_cast<uint32_t>(std::max(geometry_instance_data.size(), size_t(1))), build_as);
}

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------

bool Scene::set_build_acceleration_structure(bool build) {
    if (build_as == build)
        return false;
    build_as = build;
    rebuild_composition();
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
        scene_graph[node.parent].children.push_back(id);
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

void Scene::add_camera(CameraHandle camera) {
    cameras.push_back(std::move(camera));
}

CameraHandle Scene::get_active_camera() const {
    if (cameras.empty())
        return nullptr;
    return cameras[active_camera];
}

void Scene::set_active_camera(const uint32_t index) {
    assert(index < cameras.size());
    active_camera = index;
}

void Scene::compute_world_transforms() {
    for (auto& node : scene_graph) {
        if (node.parent == NODE_ID_INVALID) {
            node.global_transform = node.local_transform;
        } else {
            node.global_transform = scene_graph[node.parent].global_transform * node.local_transform;
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh grouping (Falcor pattern)
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
            // Non-instanced
            if (mesh.flags & GeometryFlags::IsDynamic) {
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
                // vertex/index buffer indices set during upload
                gd.vertex_buffer_index = static_cast<uint16_t>(mid);
                gd.index_buffer_index = static_cast<uint16_t>(mid);
                geometry_instance_data.push_back(gd);

                geometry_instance_id++;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry upload
// ---------------------------------------------------------------------------

void Scene::upload_geometry_buffers(const CommandBufferHandle& cmd) {
    if (!geometry_dirty)
        return;

    create_mesh_groups();

    // Upload one vertex buffer and one index buffer per mesh
    vertex_buffers.clear();
    index_buffers.clear();
    vertex_buffers.resize(meshes.size());
    index_buffers.resize(meshes.size());

    for (MeshID mid = 0; mid < static_cast<MeshID>(meshes.size()); mid++) {
        const auto& mesh = meshes[mid];
        if (mesh.instances.empty())
            continue;

        auto vb = allocator->create_buffer(
            mesh.vertices.size() * sizeof(VertexData),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            MemoryMappingType::NONE, fmt::format("Scene::vb[{}]", mid));
        allocator->get_staging()->cmd_to_device(cmd, vb, mesh.vertices.data(), 0,
                                                mesh.vertices.size() * sizeof(VertexData));
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
    }

    // Barrier: all geometry transfers → shader reads + AS build reads
    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    });

    rebuild_composition();
    geometry_dirty = false;
}

// ---------------------------------------------------------------------------
// BLAS building
// ---------------------------------------------------------------------------

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
            triangles.vertexStride = sizeof(VertexData);
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

// ---------------------------------------------------------------------------
// TLAS building
// ---------------------------------------------------------------------------

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
                const auto& t = scene_graph[nid].global_transform;
                for (int row = 0; row < 3; row++)
                    for (int col = 0; col < 4; col++)
                        inst.transform.matrix[row][col] = t[col][row]; // GLM column-major → row-major
            }

            inst.instanceCustomIndex = instance_id;
            inst.mask = 0xFF;
            inst.accelerationStructureReference = blas_list[gi]->get_acceleration_structure_device_address();

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

    auto instances_buffer = allocator->create_buffer(
        instances.size() * sizeof(vk::AccelerationStructureInstanceKHR),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        MemoryMappingType::NONE, "Scene::tlas_instances");
    allocator->get_staging()->cmd_to_device(cmd, instances_buffer, instances.data(), 0,
                                            instances.size() *
                                                sizeof(vk::AccelerationStructureInstanceKHR));

    cmd->barrier(instances_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eAccelerationStructureReadKHR));

    tlas = as_builder->queue_build(static_cast<uint32_t>(instances.size()), instances_buffer);
    as_builder->get_cmds_tlas(cmd, scratch_buffer);
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

void Scene::update(const CommandBufferHandle& cmd, const float time, const float time_diff,
                   const uint32_t frame) {
    on_update(time, time_diff);
    compute_world_transforms();

    material_system->upload(cmd);
    upload_geometry_buffers(cmd);

    if (build_as && bvh_dirty) {
        build_blas(cmd);
        build_tlas(cmd);
        bvh_dirty = false;
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
    if (geometry_data_buffer)
        c["geometries"] = geometry_data_buffer;

    c["material_system"] = material_system->get_shader_object();
    c["frame"] = frame;
    c["time"] = time;
    c["time_diff"] = time_diff;

    if (build_as && tlas) {
        c["as"] = tlas;
    }

    auto cam = get_active_camera();
    if (cam) {
        prev_active_camera.write_to(c["prev_camera"]);
        cam->write_to(c["camera"]);
        prev_active_camera = *cam;
    }
}

} // namespace merian
