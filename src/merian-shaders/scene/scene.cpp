#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/math.hpp"

#include "transform_prev_vertex.slang.spv.h"
#include "transform_vertex.slang.spv.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/format.h>
#include <unordered_map>

namespace merian {

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
    if (id >= mesh_infos.size()) {
        mesh_infos.resize(mesh_ids.size());
    }
    mesh_infos[id].mesh = std::move(mesh);

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
    node.transform_dirty = true;

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
    assert(node.is_animated);

    if (node.local_transform != local_transform) {
        node.local_transform = local_transform;

        invalidate_node(node);

        transforms_changed = true;
    }
}

void Scene::add_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_ids.is_used(mesh_id));
    assert(node_ids.is_used(node_id));

    MeshInfo& info = mesh_infos[mesh_id];

    if (info.instances.size() == 1 && (!info.is_dynamic() || pretransform_animated)) {
        info.mesh->vertices_dirty = true;
    }
    info.instances.insert(node_id);
    if (scene_graph[node_id]->is_animated)
        info.animated_instance_count++;
    needs_regroup = true;
}

void Scene::remove_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_ids.is_used(mesh_id));
    assert(node_ids.is_used(node_id));

    MeshInfo& info = mesh_infos[mesh_id];

    if (info.instances.erase(node_id) > 0) {
        if (scene_graph[node_id]->is_animated) {
            assert(info.animated_instance_count > 0);
            info.animated_instance_count--;
        }
        if (info.instances.size() == 1 && (!info.is_dynamic() || pretransform_animated)) {
            info.mesh->vertices_dirty = true;
        }
        needs_regroup = true;
    }
}

void Scene::remove_mesh(const MeshID mesh_id) {
    assert(mesh_ids.is_used(mesh_id));

    if (mesh_id < mesh_to_group.size() && mesh_to_group[mesh_id] != MESH_GROUP_ID_INVALID) {
        MeshGroup& group = mesh_groups[mesh_to_group[mesh_id]];
        group.cached_blas_size_info.reset();
        if (group.blas)
            pending_blas_releases.push_back(std::move(group.blas));
    }

    // Dropping the MeshInfo also drops its MeshBufferRegion suballocation handles, returning the
    // shared-buffer regions to the suballocator's free list.
    mesh_infos[mesh_id] = MeshInfo{};

    mesh_ids.release(mesh_id);
    mesh_infos.resize(mesh_ids.size());

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
        MeshInfo& info = mesh_infos[mesh_id];
        assert(info.mesh);
        if (info.instances.erase(node_id) > 0 && node.is_animated) {
            assert(info.animated_instance_count > 0);
            info.animated_instance_count--;
        }
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

// ---------------------------------------------------------------------------
// Batched GPU transform
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t TRANSFORM_LOCAL_SIZE_X = 256;

} // namespace

void Scene::ensure_transform_pipelines() {
    if (transform_vertex_pipeline && transform_prev_vertex_pipeline)
        return;

    transform_descriptor_layout =
        DescriptorSetLayoutBuilder()
            .add_binding_storage_buffer(1, vk::ShaderStageFlagBits::eCompute)
            .build_push_descriptor_layout(context);

    auto vertex_module = EntryPoint::create(context, merian_transform_vertex_slang_spv(),
                                            merian_transform_vertex_slang_spv_size(), "main",
                                            vk::ShaderStageFlagBits::eCompute);
    auto prev_module = EntryPoint::create(context, merian_transform_prev_vertex_slang_spv(),
                                          merian_transform_prev_vertex_slang_spv_size(), "main",
                                          vk::ShaderStageFlagBits::eCompute);

    const auto layout = PipelineLayoutBuilder(context)
                            .add_descriptor_set_layout(transform_descriptor_layout)
                            .build_pipeline_layout();

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(TRANSFORM_LOCAL_SIZE_X);
    const SpecializationInfoHandle spec = spec_builder.build();

    transform_vertex_pipeline = ComputePipeline::create(layout, vertex_module, spec);
    transform_prev_vertex_pipeline = ComputePipeline::create(layout, prev_module, spec);
}

void Scene::stage_transform_jobs(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::stage_transform_jobs");
    const auto staging = allocator->get_staging();
    const auto job_buf_usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;

    const auto ensure_job_buf = [&](BufferHandle& buf, vk::DeviceSize size, const char* name) {
        if (!buf || buf->get_size() < size) {
            if (buf)
                cmd->keep_until_pool_reset(std::move(buf));
            buf = allocator->create_buffer(size, job_buf_usage, MemoryMappingType::NONE, name);
            frame_stats.buffers_allocated++;
        }
    };

    if (!vertex_jobs.empty()) {
        const vk::DeviceSize size = vertex_jobs.size() * sizeof(TransformVertexJob);
        ensure_job_buf(transform_vertex_job_buf, size, "Scene::transform_vertex_jobs");
        staging->cmd_to_device(cmd, transform_vertex_job_buf, vertex_jobs.data(), 0, size);
        frame_stats.gpu_transform_buffer_bytes += size;
    }
    if (!prev_vertex_jobs.empty()) {
        const vk::DeviceSize size = prev_vertex_jobs.size() * sizeof(TransformPrevVertexJob);
        ensure_job_buf(transform_prev_vertex_job_buf, size, "Scene::transform_prev_vertex_jobs");
        staging->cmd_to_device(cmd, transform_prev_vertex_job_buf, prev_vertex_jobs.data(), 0,
                               size);
        frame_stats.gpu_transform_buffer_bytes += size;
    }
}

void Scene::dispatch_batched_transforms(const CommandBufferHandle& cmd) {
    ensure_transform_pipelines();

    if (!vertex_jobs.empty()) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::transform_vertices");

        const uint32_t wgx =
            (max_vertex_count_in_jobs + TRANSFORM_LOCAL_SIZE_X - 1) / TRANSFORM_LOCAL_SIZE_X;
        cmd->bind(transform_vertex_pipeline);
        cmd->push_descriptor_set(transform_vertex_pipeline,
                                 transform_vertex_job_buf->get_descriptor_info());
        cmd->dispatch(wgx, static_cast<uint32_t>(vertex_jobs.size()), 1);
    }

    if (!prev_vertex_jobs.empty()) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::transform_prev_vertices");

        const uint32_t wgx =
            (max_prev_vertex_count_in_jobs + TRANSFORM_LOCAL_SIZE_X - 1) / TRANSFORM_LOCAL_SIZE_X;
        cmd->bind(transform_prev_vertex_pipeline);
        cmd->push_descriptor_set(transform_prev_vertex_pipeline,
                                 transform_prev_vertex_job_buf->get_descriptor_info());
        cmd->dispatch(wgx, static_cast<uint32_t>(prev_vertex_jobs.size()), 1);
    }
}

void Scene::set_pretransform_animated(bool value) {
    if (pretransform_animated == value)
        return;

    pretransform_animated = value;

    needs_regroup = true;
}

void Scene::properties_node(Properties& props, const NodeID node_id) {
    if (!props.is_ui())
        return;
    if (!node_ids.is_used(node_id) || node_id >= scene_graph.size() || !scene_graph[node_id]) {
        props.output_text("<node {} not available>", node_id);
        return;
    }
    const SceneNode& node = *scene_graph[node_id];
    props.output_text("{}", node);

    for (uint32_t i = 0; i < node.children.size(); i++) {
        const NodeID child_id = node.children[i];
        const SceneNode& child = *scene_graph[child_id];
        if (props.st_begin_child(fmt::format("child_{:02}", child_id),
                                 fmt::format("Child {:04}: {}", child_id, child.name))) {
            properties_node(props, child_id);
            props.st_end_child();
        }
    }
}

void Scene::properties_mesh(Properties& props, const MeshID mesh_id) {
    if (!props.is_ui())
        return;
    if (!mesh_ids.is_used(mesh_id) || !mesh_infos[mesh_id].mesh) {
        props.output_text("<mesh {} not available>", mesh_id);
        return;
    }
    const MeshInfo& info = mesh_infos[mesh_id];
    const Mesh& mesh = *info.mesh;
    const MeshGroupID group_id =
        mesh_id < mesh_to_group.size() ? mesh_to_group[mesh_id] : MESH_GROUP_ID_INVALID;

    props.output_text("{}", mesh);

    if (group_id != MESH_GROUP_ID_INVALID &&
        props.st_begin_child("group", fmt::format("group {}", group_id))) {
        properties_mesh_group(props, group_id);
        props.st_end_child();
    }

    if (props.st_begin_child("instances",
                             fmt::format("instances ({})", info.instances.size()))) {
        for (const NodeID node_id : info.instances) {
            const std::string node_name = (node_id < scene_graph.size() && scene_graph[node_id])
                                              ? scene_graph[node_id]->name
                                              : std::string{"<dead>"};
            if (props.st_begin_child(fmt::format("instance_{:04}", node_id),
                                     fmt::format("{:04}: {}", node_id, node_name))) {
                properties_node(props, node_id);
                props.st_end_child();
            }
        }
        props.st_end_child();
    }
}

void Scene::properties_mesh_group(Properties& props, const MeshGroupID group_id) {
    if (!props.is_ui())
        return;
    if (group_id >= mesh_groups.size()) {
        props.output_text("<group {} not available>", group_id);
        return;
    }
    const MeshGroup& group = mesh_groups[group_id];
    const std::size_t instances =
        group.meshes.empty() ? 0 : group.get_instances(mesh_infos).size();
    const bool pretransformed =
        !group.meshes.empty() && group.is_pretranformed(mesh_infos, pretransform_animated);

    std::size_t group_vertices = 0;
    std::size_t group_triangles = 0;
    for (const MeshID mesh_id : group.meshes) {
        group_vertices += mesh_infos[mesh_id].mesh->get_vertex_count();
        group_triangles += mesh_infos[mesh_id].mesh->get_primitive_count();
    }
    props.output_text(
        "split flags: {}\nhas animated node: {}\nhas morphed mesh: {}\nhas variable topology: "
        "{}\nall opaque: {}\npretransformed: {}\ninstances: {}\nvertices: {}\ntriangles: "
        "{}\nblas: {}\nblas build flags: {}\nblas dirty: {}\nblas last built frame: {}\nblas "
        "last updated frame: {}",
        group.flags, group.has_animated_node, group.has_morphed_mesh,
        group.has_variable_topology_mesh, group.all_opaque, pretransformed, instances,
        group_vertices, group_triangles,
        group.blas ? format_size(group.blas->get_size()) : "<none>",
        vk::to_string(group.blas_build_flags), group.blas_dirty, group.blas_last_built_frame,
        group.blas_last_updated_frame);

    if (props.st_begin_child("members", fmt::format("members ({})", group.meshes.size()))) {
        for (const MeshID mesh_id : group.meshes) {
            const std::string mesh_name = mesh_infos[mesh_id].mesh
                                              ? mesh_infos[mesh_id].mesh->name
                                              : std::string{"<dead>"};
            if (props.st_begin_child(fmt::format("mesh_{:04}", mesh_id),
                                     fmt::format("{:04}: {}", mesh_id, mesh_name))) {
                properties_mesh(props, mesh_id);
                props.st_end_child();
            }
        }
        props.st_end_child();
    }
}

void Scene::properties_cameras(Properties& props) {
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
            auto db = std::make_shared<Camera>(float3(1, 0, 0), float3(0, 0, 0), get_up(), 90.f,
                                               1920.f / 1080.f, 0.01f, 1000.f);
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
}

void Scene::properties_meshes(Properties& props) {
    if (!props.is_ui())
        return;
    for (const MeshID id : mesh_ids) {
        const MeshInfo& info = mesh_infos[id];
        const Mesh& mesh = *info.mesh;
        const MeshGroupID group_id =
            id < mesh_to_group.size() ? mesh_to_group[id] : MESH_GROUP_ID_INVALID;
        const std::string group_label = group_id == MESH_GROUP_ID_INVALID
                                            ? std::string{"no group"}
                                            : fmt::format("group {}", group_id);
        if (props.st_begin_child(fmt::format("mesh_{:04}", id),
                                 fmt::format("{:04}: {} [{}, {} instances]", id, mesh.name,
                                             group_label, info.instances.size()))) {
            properties_mesh(props, id);
            props.st_end_child();
        }
    }
}

void Scene::properties_mesh_groups(Properties& props) {
    if (!props.is_ui())
        return;
    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        const MeshGroup& group = mesh_groups[group_id];
        // When the group holds a single mesh, surface that mesh inline so the
        // explorer is navigable without expanding every row.
        std::string label;
        if (group.meshes.size() == 1) {
            const MeshID only = *group.meshes.begin();
            const std::string mesh_name = mesh_infos[only].mesh
                                              ? mesh_infos[only].mesh->name
                                              : std::string{"<dead>"};
            label = fmt::format("{:04}: {} (1 mesh, {:04}: {})", group_id, group.flags, only,
                                mesh_name);
        } else {
            label = fmt::format("{:04}: {} ({} meshes)", group_id, group.flags,
                                group.meshes.size());
        }
        if (props.st_begin_child(fmt::format("group_{:04}", group_id), label)) {
            properties_mesh_group(props, group_id);
            props.st_end_child();
        }
    }
}

void Scene::properties_graph(Properties& props) {
    if (!props.is_ui())
        return;
    for (const NodeID id : node_ids) {
        const SceneNode& node = *scene_graph[id];
        if (node.parent != NODE_ID_INVALID) {
            continue;
        }
        if (props.st_begin_child(fmt::format("node_{:04}", id),
                                 fmt::format("{:04}: {}", id, node.name))) {
            properties_node(props, id);
            props.st_end_child();
        }
    }
}

void Scene::properties_explorer(Properties& props) {
    if (props.st_begin_child("cameras", "Cameras")) {
        properties_cameras(props);
        props.st_end_child();
    }
    if (!props.is_ui())
        return;
    if (props.st_begin_child("meshes", fmt::format("Meshes ({})", mesh_ids.count()))) {
        properties_meshes(props);
        props.st_end_child();
    }
    if (props.st_begin_child("mesh_groups",
                             fmt::format("Mesh Groups ({})", mesh_groups.size()))) {
        properties_mesh_groups(props);
        props.st_end_child();
    }
    if (props.st_begin_child("graph", fmt::format("Graph ({} nodes)", node_ids.count()))) {
        properties_graph(props);
        props.st_end_child();
    }
}

void Scene::properties_settings(Properties& props) {
    bool pretransform = get_pretransform_animated();
    if (props.config_bool("Pretransform Animated", pretransform)) {
        set_pretransform_animated(pretransform);
    }
    props.config_percent("BLAS Rebuild Fraction", blas_rebuild_fraction);
}

void Scene::properties(Properties& props) {

    if (props.st_begin_child("scene", "Explorer")) {
        properties_explorer(props);
        props.st_end_child();
    }

    if (props.st_begin_child("settings", "Settings")) {
        properties_settings(props);
        props.st_end_child();
    }

    if (props.st_begin_child("stats", "Statistics")) {
        std::size_t total_vertices = 0;
        std::size_t total_triangles = 0;
        std::size_t total_instances = 0;
        std::size_t morphed_meshes = 0;
        std::size_t dynamic_meshes = 0;
        std::size_t animated_instance_meshes = 0;
        std::size_t variable_topology_meshes = 0;
        std::size_t animated_instances = 0;
        for (const MeshID id : mesh_ids) {
            const MeshInfo& info = mesh_infos[id];
            const Mesh& mesh = *info.mesh;
            total_vertices += mesh.get_vertex_count();
            total_triangles += mesh.get_primitive_count();
            total_instances += info.instances.size();
            animated_instances += info.animated_instance_count;
            if (mesh.is_morphed())
                morphed_meshes++;
            if (info.is_dynamic())
                dynamic_meshes++;
            if (info.animated_instance_count > 0)
                animated_instance_meshes++;
            if (mesh.has_variable_topology())
                variable_topology_meshes++;
        }

        props.output_text(
            "nodes:       {}\n"
            "meshes:      {} (dynamic: {}, morphed: {}, animated: {}, var_topo: {})\n"
            "instances:   {} (animated: {})\n"
            "mesh groups: {}\n"
            "vertices:    {}\n"
            "triangles:   {}\n"
            "materials:   {}\n"
            "textures:    {}\n"
            "tlas:        {} instances\n"
            "flags:       needs_regroup={}, transforms_changed={}, pretransform_animated={}\n"
            "pending buffer releases: {}",
            node_ids.count(), mesh_ids.count(), dynamic_meshes, morphed_meshes,
            animated_instance_meshes, variable_topology_meshes, total_instances, animated_instances,
            mesh_groups.size(), total_vertices, total_triangles,
            material_system->get_material_count(), get_texture_manager()->get_texture_count(),
            tlas_instances.size(), needs_regroup, transforms_changed, pretransform_animated,
            pending_buffer_releases.size());

        if (aabb.is_valid()) {
            props.output_text("aabb: min={}, max={}, size={}", aabb.get_min(), aabb.get_max(),
                              aabb.get_max() - aabb.get_min());
        } else {
            props.output_text("aabb: <not available>");
        }

        props.st_separate("Per Frame");
        props.output_text(
            "meshes uploaded: {} (device_local: {}, device_staged: {}, host_packed: {}, "
            "host_unpacked: {})\n"
            "upload:          {} ({} vertices, {} primitives)\n"
            "vertex xfm:      {} jobs ({} vertices)\n"
            "prev vertex xfm: {} jobs ({} vertices)\n"
            "xfm job buffers: {}\n"
            "blas:            {} ops (builds_static: {}, builds_dynamic: {}, updates: {})\n"
            "tlas:            {} ({} instances)\n"
            "buffers:         allocated: {}, released: {}\n"
            "gpu data:        geometry {}, transforms {}, tlas instances {}",
            frame_stats.meshes_uploaded(), frame_stats.meshes_uploaded_device_local,
            frame_stats.meshes_uploaded_device_staged, frame_stats.meshes_uploaded_host_packed,
            frame_stats.meshes_uploaded_host_unpacked, format_size(frame_stats.upload_bytes),
            frame_stats.vertices_uploaded, frame_stats.indices_uploaded,
            frame_stats.gpu_vertex_transforms, frame_stats.gpu_vertex_transform_vertices,
            frame_stats.gpu_prev_vertex_transforms, frame_stats.gpu_prev_vertex_transform_vertices,
            format_size(frame_stats.gpu_transform_buffer_bytes),
            frame_stats.blas_builds + frame_stats.blas_updates, frame_stats.blas_builds_static,
            frame_stats.blas_builds_dynamic, frame_stats.blas_updates,
            frame_stats.tlas_rebuilt ? "rebuilt" : "unchanged", frame_stats.tlas_instance_count,
            frame_stats.buffers_allocated, frame_stats.buffers_released,
            format_size(frame_stats.geometry_data_bytes),
            format_size(frame_stats.transform_data_bytes),
            format_size(frame_stats.tlas_instance_data_bytes));

        props.st_separate("Shared Buffers");
        const auto block_text = [&](const char* name, const VMAMemorySubAllocatorHandle& slot,
                                    const vk::DeviceSize capacity) {
            if (!slot) {
                props.output_text("{:<14} <not allocated>", name);
                return;
            }
            const VmaDetailedStatistics s = slot->get_detailed_statistics();
            const auto fmt_size = [](const vk::DeviceSize x) {
                return x == VK_WHOLE_SIZE ? std::string{"-"} : format_size(x);
            };
            const double fill = s.statistics.blockBytes == 0
                                    ? 0.0
                                    : 100.0 * static_cast<double>(s.statistics.allocationBytes) /
                                          static_cast<double>(s.statistics.blockBytes);
            props.output_text("{:<14} {} / {} ({:.1f}% fill) — allocs: {} ({}..{}), free: {} "
                              "ranges ({}..{})",
                              fmt::format("{}:", name), format_size(s.statistics.allocationBytes),
                              format_size(capacity), fill, s.statistics.allocationCount,
                              fmt_size(s.allocationSizeMin), fmt_size(s.allocationSizeMax),
                              s.unusedRangeCount, fmt_size(s.unusedRangeSizeMin),
                              fmt_size(s.unusedRangeSizeMax));
        };
        block_text("shared_vb", shared_vb_suballoc, shared_vb_capacity);
        block_text("shared_prev_vb", shared_prev_vb_suballoc, shared_prev_vb_capacity);
        block_text("shared_ib", shared_ib_suballoc, shared_ib_capacity);
        if (frame_staging) {
            props.output_text("{:<14} {} (bump-allocated, persistently mapped)",
                              "frame_staging:", format_size(frame_staging->get_capacity()));
        } else {
            props.output_text("{:<14} <not allocated>", "frame_staging:");
        }

        props.st_end_child();
    }
}

// ---------------------------------------------------------------------------
// Scene update
// ---------------------------------------------------------------------------

void Scene::compute_mesh_groups() {
    MERIAN_PROFILE_SCOPE("Scene::compute_mesh_groups");

    auto prev_mesh_groups = std::move(mesh_groups);
    auto prev_mesh_to_group = std::move(mesh_to_group);
    mesh_groups.clear();
    mesh_to_group.assign(mesh_ids.size(), MESH_GROUP_ID_INVALID);

    // allow to access prev_mesh_to_group with any (new) MeshID.
    prev_mesh_to_group.resize(mesh_ids.size(), MESH_GROUP_ID_INVALID);
    mesh_groups.reserve(std::min<std::size_t>(mesh_ids.count(), prev_mesh_groups.size()));

    // 1. Group meshes according to our grouping logic (see scene.hpp)
    // Only FrontCounterClockwise and TwoSided split groups (TLAS instance flags).
    // IsOpaque is per-geometry, IsMorphed is per-mesh.

    const auto split_key = [](MeshFlags f) -> uint32_t {
        return static_cast<uint32_t>(f) & static_cast<uint32_t>(GROUP_SPLIT_MASK);
    };

    // static (non-morphed, non-animated): pretransformed, BLAS built once
    std::unordered_map<uint32_t, MeshGroupID> groups_static;
    // dynamic with pretransform_animated: pretransformed, BLAS rebuilt as needed
    std::unordered_map<uint32_t, MeshGroupID> groups_dynamic_pretransformed;
    // morphed on non-animated node, pretransform OFF: not pretransformed
    std::unordered_map<uint32_t, MeshGroupID> groups_morphed_static;
    // animated node, pretransform OFF: not pretransformed, per-node
    std::unordered_map<uint64_t, MeshGroupID> groups_animated_per_node;
    // instanced
    std::unordered_map<SmallSet<NodeID, 1>, std::unordered_map<uint32_t, MeshGroupID>>
        groups_instanced;

    groups_static.reserve(32);
    groups_animated_per_node.reserve(node_ids.count() * 2);

    const auto get_or_create = [&](auto& map, auto key) -> MeshGroupID {
        const auto [it, inserted] = map.try_emplace(key, mesh_groups.size());
        if (inserted) {
            mesh_groups.emplace_back();
        }
        return it->second;
    };

    for (const MeshID mesh_id : mesh_ids) {
        const MeshInfo& info = mesh_infos[mesh_id];
        const Mesh& mesh = *info.mesh;

        if (info.instances.empty()) {
            continue;
        }

        const uint32_t key = split_key(mesh.flags);
        MeshGroupID group_id;

        if (info.instances.size() > 1) {
            group_id = get_or_create(groups_instanced[info.instances], key);
        } else {
            assert(info.instances.size() == 1);
            const NodeID node_id = *info.instances.begin();
            const bool node_animated = scene_graph[node_id]->is_animated;
            const bool is_dynamic = node_animated || mesh.is_morphed();

            if (!is_dynamic) {
                group_id = get_or_create(groups_static, key);
            } else if (pretransform_animated) {
                group_id = get_or_create(groups_dynamic_pretransformed, key);
            } else if (!node_animated) {
                group_id = get_or_create(groups_morphed_static, key);
            } else {
                const uint64_t node_key = (static_cast<uint64_t>(key) << 32) | node_id;
                group_id = get_or_create(groups_animated_per_node, node_key);
            }
        }

        mesh_to_group[mesh_id] = group_id;
        MeshGroup& group = mesh_groups[group_id];
        group.flags = static_cast<MeshFlags>(key);
        group.has_animated_node |= info.animated_instance_count > 0;
        group.has_morphed_mesh |= mesh.is_morphed();
        group.has_variable_topology_mesh |= mesh.has_variable_topology();
        group.all_opaque &= mesh.is_opaque();
        group.meshes.insert(mesh_id);
    }

    // 2. Compute build flags and try to find previous BLASs to prevent (re)builds.
    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        // BLAS-space vertices change iff the mesh is morphed, or pretransform_animated
        // bakes a changing node transform into the vertices. An animated node alone
        // (with pretransform_animated off) only moves the TLAS instance, so the BLAS
        // is static and needs neither rebuild nor AllowUpdate.
        const bool blas_vertices_change =
            group.has_morphed_mesh || (group.has_animated_node && pretransform_animated);
        if (!blas_vertices_change) {
            group.blas_build_flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
        } else if (!group.has_variable_topology_mesh) {
            group.blas_build_flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                                     vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
        } else {
            group.blas_build_flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
        }

        // We can check the first mesh only, since if thats not in the old group we cant reuse the
        // group / BLAS anyway.
        MeshGroupID maybe_prev_group = prev_mesh_to_group[*group.meshes.begin()];
        if (maybe_prev_group != MESH_GROUP_ID_INVALID) {
            MeshGroup& prev_group = prev_mesh_groups[maybe_prev_group];
            if (prev_group.meshes == group.meshes) {
                group.blas = prev_group.blas;
                group.cached_blas_size_info = prev_group.cached_blas_size_info;
                // blas_dirty is computed later when we upload the meshes, because this method is
                // not run every frame. We later also check if the blas can be actually reused or
                // if a new one is necessary.
            }
        }
    }

    for (auto& prev_group : prev_mesh_groups) {
        if (prev_group.blas)
            pending_blas_releases.push_back(std::move(prev_group.blas));
    }
}

void Scene::upload_transforms(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::upload_transforms");
    instance_transforms.clear();
    inverse_transposed_instance_transforms.clear();
    prev_instance_transforms.clear();
    prev_inverse_transposed_instance_transforms.clear();

    const float4x4 identity_transform = identity();

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        const bool pretransform = group.is_pretranformed(mesh_infos, pretransform_animated);

        for (const NodeID node_id : group.get_instances(mesh_infos)) {
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
}

void Scene::upload_geometry_data(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::upload_geometry_data");
    geometries.clear();

    const float4x4 identity_transform = identity();

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        const bool pretransform = group.is_pretranformed(mesh_infos, pretransform_animated);

        for (const NodeID node_id : group.get_instances(mesh_infos)) {
            SceneNode& node = *scene_graph[node_id];
            const bool transform_is_identity = get_global_transform(node) == identity_transform;

            for (const MeshID mesh_id : group.meshes) {
                MeshInfo& info = mesh_infos[mesh_id];
                Mesh& mesh = *info.mesh;
                assert(info.vertex_buffer && (!mesh.has_indices() || info.index_buffer));

                const bool needs_prev = mesh.is_morphed() || mesh.has_variable_topology() ||
                                        (pretransform && scene_graph[node_id]->is_animated);

                GeometryData gd;
                gd.material_id = mesh.material_id;
                gd.vertices = info.vertex_buffer.get_device_address();
                gd.indices = info.index_buffer ? info.index_buffer.get_device_address()
                                               : vk::DeviceAddress{0};
                assert(!needs_prev || info.prev_vertex_buffer);
                gd.prev_vertices = info.prev_vertex_buffer
                                       ? info.prev_vertex_buffer.get_device_address()
                                       : vk::DeviceAddress{0};
                gd.flags = index_type_flag(mesh.index_type);
                if (pretransform || transform_is_identity) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::Pretransformed);
                }
                if (needs_prev) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::HasPrevVertices);
                }
                if (mesh.is_front_counterclockwise()) {
                    gd.flags =
                        GeometryDataFlags(gd.flags | GeometryDataFlags::FrontCounterClockwise);
                }
                if (scene_graph[node_id]->is_animated) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::IsAnimated);
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

namespace {

constexpr vk::BufferUsageFlags SHARED_VB_USAGE =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
    vk::BufferUsageFlagBits::eShaderDeviceAddress;

constexpr vk::BufferUsageFlags SHARED_IB_USAGE =
    SHARED_VB_USAGE | vk::BufferUsageFlagBits::eIndexBuffer;

constexpr vk::BufferUsageFlags STAGING_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress;

constexpr vk::DeviceSize MIN_BUFFER_CAPACITY = vk::DeviceSize(32) * 1024 * 1024;
constexpr vk::DeviceSize RESIZE_QUANTUM = vk::DeviceSize(32) * 1024 * 1024;
constexpr vk::DeviceSize MIN_FREE_HEADROOM = vk::DeviceSize(16) * 1024 * 1024;

constexpr vk::DeviceSize size_with_headroom(const vk::DeviceSize required) {
    const vk::DeviceSize free_space = std::max(MIN_FREE_HEADROOM, required / 4);
    const vk::DeviceSize target = std::max(MIN_BUFFER_CAPACITY, required + free_space);
    return ((target + RESIZE_QUANTUM - 1) / RESIZE_QUANTUM) * RESIZE_QUANTUM;
}

} // namespace

void Scene::reallocate_shared_buffer(const CommandBufferHandle& cmd,
                                     VMAMemorySubAllocatorHandle& slot,
                                     vk::DeviceSize& capacity_slot,
                                     MeshBufferRegion MeshInfo::* region_field,
                                     const vk::DeviceSize new_capacity,
                                     const vk::BufferUsageFlags usage,
                                     const std::string& debug_name) {
    const vk::DeviceSize old_capacity = capacity_slot;
    const BufferHandle new_buffer = allocator->create_buffer(
        vk::BufferCreateInfo{{}, new_capacity, usage}, MemoryMappingType::NONE, debug_name);
    const VMAMemorySubAllocatorHandle new_alloc = VMAMemorySubAllocator::create(new_buffer);
    frame_stats.buffers_allocated++;

    const BufferHandle old_buffer = slot ? slot->get_base_buffer() : nullptr;
    std::vector<vk::BufferCopy> regions;
    regions.reserve(mesh_ids.count());

    for (const MeshID id : mesh_ids) {
        MeshBufferRegion& region = mesh_infos[id].*region_field;
        if (!region)
            continue;

        const vk::DeviceSize old_offset = region.get_offset();
        const auto va = new_alloc->allocate({region.size, region.alignment, ~0u});
        regions.push_back({old_offset, va.offset, region.size});
        region.suballoc = std::make_shared<VMAMemorySubAllocation>(
            context, new_alloc, va.allocation, va.offset, region.size);
    }

    if (old_buffer && !regions.empty()) {
        cmd->copy(old_buffer, new_buffer, regions);
    }

    slot = new_alloc;
    capacity_slot = new_capacity;

    SPDLOG_DEBUG("Scene::reallocate_shared_buffer '{}': {} -> {}, migrated {} regions{}",
                 debug_name, format_size(old_capacity), format_size(new_capacity), regions.size(),
                 old_buffer ? "" : " (initial)");
}

void Scene::upload_meshes(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::upload_meshes");

    vertex_jobs.clear();
    prev_vertex_jobs.clear();
    vertex_jobs.reserve(mesh_ids.count());
    prev_vertex_jobs.reserve(mesh_ids.count());
    max_vertex_count_in_jobs = 0;
    max_prev_vertex_count_in_jobs = 0;

    vk::DeviceSize total_vb = 0;
    vk::DeviceSize total_prev_vb = 0;
    vk::DeviceSize total_ib = 0;
    {
        MERIAN_PROFILE_SCOPE("size_prepass");
        for (const MeshID id : mesh_ids) {
            const MeshInfo& info = mesh_infos[id];
            if (info.instances.empty())
                continue;
            const Mesh& mesh = *info.mesh;
            const uint32_t vc = mesh.get_vertex_count();
            const uint32_t pc = mesh.get_primitive_count();
            total_vb += vc * sizeof(PackedVertexData);
            total_ib += pc * std::size_t(3) * size_for_index_type(mesh.index_type);
            if (mesh.is_morphed() || mesh.has_variable_topology() || info.is_dynamic())
                total_prev_vb += vc * sizeof(PackedPrevVertexData);
        }
    }

    if (total_vb > shared_vb_capacity) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "realloc_vb");
        reallocate_shared_buffer(cmd, shared_vb_suballoc, shared_vb_capacity,
                                 &MeshInfo::vertex_buffer, size_with_headroom(total_vb),
                                 SHARED_VB_USAGE, "Scene::shared_vb");
    }
    if (total_prev_vb > shared_prev_vb_capacity) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "realloc_prev_vb");
        reallocate_shared_buffer(cmd, shared_prev_vb_suballoc, shared_prev_vb_capacity,
                                 &MeshInfo::prev_vertex_buffer, size_with_headroom(total_prev_vb),
                                 SHARED_VB_USAGE, "Scene::shared_prev_vb");
    }
    if (total_ib > shared_ib_capacity) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "realloc_ib");
        reallocate_shared_buffer(cmd, shared_ib_suballoc, shared_ib_capacity,
                                 &MeshInfo::index_buffer, size_with_headroom(total_ib),
                                 SHARED_IB_USAGE, "Scene::shared_ib");
    }

    if (!frame_staging) {
        frame_staging.emplace(allocator->get_memory_allocator(), STAGING_USAGE, MIN_BUFFER_CAPACITY,
                              "Scene::frame_staging");
        frame_stats.buffers_allocated++;
    }
    frame_staging->reset();

    // No-op if region already matches; reused regions stay where they are.
    const auto ensure_region = [&](MeshBufferRegion& region,
                                   const VMAMemorySubAllocatorHandle& slot,
                                   const vk::DeviceSize size, const vk::DeviceSize alignment) {
        if (region && region.size == size && region.alignment == alignment)
            return;
        region.suballoc.reset();
        const auto va = slot->allocate({size, alignment, ~0u});
        region.suballoc =
            std::make_shared<VMAMemorySubAllocation>(context, slot, va.allocation, va.offset, size);
        region.size = size;
        region.alignment = alignment;
    };

    const auto staging_alloc = [&](const vk::DeviceSize size,
                                   const vk::DeviceSize align) -> FrameStagingBlock::Slice {
        if (auto slice = frame_staging->alloc(size, align)) {
            return *slice;
        }
        frame_staging->grow(cmd, size_with_headroom(frame_staging->get_capacity() + size));
        frame_stats.buffers_allocated++;
        const auto retried = frame_staging->alloc(size, align);
        assert(retried.has_value() && "frame_staging grow failed to satisfy alloc");
        return *retried;
    };

    // Batch one cmd->copy per distinct source buffer.
    std::unordered_map<Buffer*, std::pair<BufferHandle, std::vector<vk::BufferCopy>>> index_copies;

    bool did_upload_static = false;

    {
        MERIAN_PROFILE_SCOPE("fill_staging");
        for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
            MeshGroup& group = mesh_groups[group_id];
            const bool pretransform_group =
                group.is_pretranformed(mesh_infos, pretransform_animated);
            // Pretransformed animated meshes must be re-pretransformed when the node moved.
            const bool check_node_transform = pretransform_group && group.has_animated_node;

            for (const MeshID mesh_id : group.meshes) {
                MeshInfo& info = mesh_infos[mesh_id];
                Mesh& mesh = *info.mesh;

                mesh.vertices_dirty |=
                    check_node_transform && scene_graph[*info.instances.begin()]->transform_dirty;
                if (!mesh.is_dirty())
                    continue;

                if (!info.is_dynamic())
                    did_upload_static = true;

                group.blas_dirty = true;
                if (mesh.has_variable_topology())
                    group.cached_blas_size_info.reset();

                const uint32_t vertex_count = mesh.get_vertex_count();
                const uint32_t primitive_count = mesh.get_primitive_count();
                const vk::DeviceSize vb_size = vertex_count * sizeof(PackedVertexData);
                const vk::DeviceSize prev_vb_size = vertex_count * sizeof(PackedPrevVertexData);
                const vk::DeviceSize ib_size =
                    primitive_count * std::size_t(3) * size_for_index_type(mesh.index_type);

                frame_stats.vertices_uploaded += vertex_count;
                frame_stats.indices_uploaded += primitive_count;
                frame_stats.upload_bytes += vb_size + ib_size;

                const NodeID node_id = *info.instances.begin();
                const float4x4& M = get_global_transform(node_id);
                const float4x4& M_it = get_global_inverse_transposed_transform(node_id);
                const bool pretransform_mesh = pretransform_group && M != identity();
                const bool synth_prev = pretransform_mesh && !mesh.is_morphed() &&
                                        !mesh.has_variable_topology() &&
                                        scene_graph[node_id]->is_animated;

                ensure_region(info.vertex_buffer, shared_vb_suballoc, vb_size,
                              alignof(PackedVertexData));

                const bool need_prev_buffer =
                    mesh.is_morphed() || mesh.has_variable_topology() || synth_prev;
                if (need_prev_buffer) {
                    ensure_region(info.prev_vertex_buffer, shared_prev_vb_suballoc, prev_vb_size,
                                  alignof(PackedPrevVertexData));
                } else if (info.prev_vertex_buffer) {
                    info.prev_vertex_buffer = MeshBufferRegion{};
                }

                if (mesh.vertices_dirty) {
                    vk::DeviceAddress vertex_src_addr = 0;
                    std::visit(
                        [&](auto&& src) {
                            using T = std::decay_t<decltype(src)>;
                            if constexpr (std::is_same_v<T, Mesh::DeviceLocal>) {
                                frame_stats.meshes_uploaded_device_local++;
                                vertex_src_addr = src.data->get_device_address();
                            } else if constexpr (std::is_same_v<T, Mesh::DeviceStaged>) {
                                frame_stats.meshes_uploaded_device_staged++;
                                vertex_src_addr = src.data->get_device_address() + src.offset;
                            } else if constexpr (std::is_same_v<
                                                     T, Mesh::HostPacked<PackedVertexData>>) {
                                frame_stats.meshes_uploaded_host_packed++;
                                const auto slice =
                                    staging_alloc(vb_size, alignof(PackedVertexData));
                                vertex_src_addr = slice.addr;
                                std::memcpy(slice.ptr, src.data, vb_size);
                            } else {
                                static_assert(std::is_same_v<T, Mesh::HostVertices>);
                                frame_stats.meshes_uploaded_host_unpacked++;
                                const auto slice =
                                    staging_alloc(vb_size, alignof(PackedVertexData));
                                vertex_src_addr = slice.addr;
                                src->write(reinterpret_cast<PackedVertexData*>(slice.ptr));
                            }
                        },
                        mesh.get_vertices());

                    if (need_prev_buffer && !synth_prev) {
                        vk::DeviceAddress prev_src_addr = 0;
                        std::visit(
                            [&](auto&& src) {
                                using T = std::decay_t<decltype(src)>;
                                if constexpr (std::is_same_v<T, std::monostate>) {
                                    assert(false);
                                } else if constexpr (std::is_same_v<T, Mesh::DeviceLocal>) {
                                    prev_src_addr = src.data->get_device_address();
                                } else if constexpr (std::is_same_v<T, Mesh::DeviceStaged>) {
                                    prev_src_addr = src.data->get_device_address() + src.offset;
                                } else if constexpr (std::is_same_v<T, Mesh::HostPacked<
                                                                           PackedPrevVertexData>>) {
                                    const auto slice =
                                        staging_alloc(prev_vb_size, alignof(PackedPrevVertexData));
                                    prev_src_addr = slice.addr;
                                    std::memcpy(slice.ptr, src.data, prev_vb_size);
                                } else {
                                    static_assert(std::is_same_v<T, Mesh::HostPrevVertices>);
                                    const auto slice =
                                        staging_alloc(prev_vb_size, alignof(PackedPrevVertexData));
                                    prev_src_addr = slice.addr;
                                    src->write(reinterpret_cast<PackedPrevVertexData*>(slice.ptr));
                                }
                            },
                            mesh.get_prev_vertices());

                        const float4x4 prev_M =
                            pretransform_mesh
                                ? scene_graph[node_id]->prev_global_transform.value_or(M)
                                : float4x4(identity());
                        prev_vertex_jobs.push_back(TransformPrevVertexJob{
                            .src = prev_src_addr,
                            .dst = info.prev_vertex_buffer.get_device_address(),
                            .prev_transform = prev_M,
                            .vertex_count = vertex_count,
                            ._pad = {},
                        });
                        max_prev_vertex_count_in_jobs =
                            std::max(max_prev_vertex_count_in_jobs, vertex_count);
                        frame_stats.gpu_prev_vertex_transforms++;
                        frame_stats.gpu_prev_vertex_transform_vertices += vertex_count;
                        frame_stats.upload_bytes += prev_vb_size;
                    }

                    const float4x4 prev_M =
                        synth_prev ? scene_graph[node_id]->prev_global_transform.value_or(M)
                                   : float4x4(identity());
                    vertex_jobs.push_back(TransformVertexJob{
                        .src = vertex_src_addr,
                        .dst = info.vertex_buffer.get_device_address(),
                        .prev_dst = synth_prev ? info.prev_vertex_buffer.get_device_address() : 0,
                        .transform = pretransform_mesh ? M : float4x4(identity()),
                        .inverse_transposed = pretransform_mesh ? M_it : float4x4(identity()),
                        .prev_transform = prev_M,
                        .vertex_count = vertex_count,
                        ._pad = {},
                    });
                    max_vertex_count_in_jobs = std::max(max_vertex_count_in_jobs, vertex_count);
                    frame_stats.gpu_vertex_transforms++;
                    frame_stats.gpu_vertex_transform_vertices += vertex_count;

                    mesh.vertices_dirty = false;
                }

                if (mesh.indices_dirty && mesh.has_indices()) {
                    ensure_region(info.index_buffer, shared_ib_suballoc, ib_size,
                                  size_for_index_type(mesh.index_type));
                    const vk::DeviceSize dst_offset = info.index_buffer.get_offset();

                    std::visit(
                        [&](auto&& src) {
                            using T = std::decay_t<decltype(src)>;
                            if constexpr (std::is_same_v<T, std::monostate>) {
                                assert(false && "monostate indices on a mesh with has_indices()");
                            } else if constexpr (std::is_same_v<T, Mesh::DeviceLocal>) {
                                auto& entry = index_copies[src.data.get()];
                                entry.first = src.data;
                                entry.second.push_back({0, dst_offset, ib_size});
                            } else if constexpr (std::is_same_v<T, Mesh::DeviceStaged>) {
                                auto& entry = index_copies[src.data.get()];
                                entry.first = src.data;
                                entry.second.push_back({src.offset, dst_offset, ib_size});
                            } else if constexpr (std::is_same_v<T, Mesh::HostPacked<void>>) {
                                const auto slice =
                                    staging_alloc(ib_size, size_for_index_type(mesh.index_type));
                                std::memcpy(slice.ptr, src.data, ib_size);
                                auto& entry = index_copies[slice.buffer.get()];
                                entry.first = slice.buffer;
                                entry.second.push_back({slice.buffer_offset, dst_offset, ib_size});
                            } else {
                                static_assert(std::is_same_v<T, Mesh::HostIndices>);
                                const auto slice =
                                    staging_alloc(ib_size, size_for_index_type(mesh.index_type));
                                src->write(slice.ptr);
                                auto& entry = index_copies[slice.buffer.get()];
                                entry.first = slice.buffer;
                                entry.second.push_back({slice.buffer_offset, dst_offset, ib_size});
                            }
                        },
                        mesh.get_indices());

                    mesh.indices_dirty = false;
                } else if (mesh.indices_dirty) {
                    // No index buffer for IndexType::None; drop any prior allocation.
                    if (info.index_buffer)
                        info.index_buffer = MeshBufferRegion{};
                    mesh.indices_dirty = false;
                }
            }
        }
    }

    if (!index_copies.empty()) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "index_copy");
        // Source buffers (Quake's brush_ib etc., or our own frame_staging) may have been written
        // earlier in the same cmd via cmd_to_device / updateBuffer. Sync transfer-write -> read.
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eHost,
            vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostWrite,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferRead,
        });

        const BufferHandle& dst = shared_ib_suballoc->get_base_buffer();
        for (auto& [_, entry] : index_copies) {
            cmd->copy(entry.first, dst, entry.second);
        }
    }

    if (did_upload_static && frame_staging->get_capacity() > MIN_BUFFER_CAPACITY) {
        frame_staging.reset();
    }
}

void Scene::build_blas(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::build_blas");

    blas_geometries.assign(mesh_groups.size(), {});

    bool did_build_static = false;
    std::vector<MeshGroupID> update_eligible;

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        if (group.blas && !group.blas_dirty) {
            continue;
        }
        tlas_dirty = true;

        auto& blas_geometry = blas_geometries[group_id];

        for (MeshID mesh_id : group.meshes) {
            const MeshInfo& info = mesh_infos[mesh_id];
            const Mesh& mesh = *info.mesh;

            vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
            triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
            triangles.vertexData = info.vertex_buffer.get_device_address();
            triangles.vertexStride = sizeof(PackedVertexData);
            triangles.maxVertex = mesh.get_vertex_count() - 1;
            triangles.indexType = mesh.index_type;
            triangles.indexData = info.index_buffer ? info.index_buffer.get_device_address()
                                                    : vk::DeviceAddress{0};

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

        const bool can_update =
            group.blas &&
            (group.blas_build_flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate);

        if (can_update) {
            update_eligible.push_back(group_id);
            continue;
        }

        const bool static_geometry = !group.has_morphed_mesh && !group.has_animated_node;
        if (static_geometry)
            did_build_static = true;

        if (!group.cached_blas_size_info) {
            group.cached_blas_size_info = as_builder.get_size_info(
                blas_geometry.geometries, blas_geometry.ranges, group.blas_build_flags);
        }
        const auto& size_info = *group.cached_blas_size_info;

        if (!group.blas || group.blas->get_size() < size_info.accelerationStructureSize) {
            group.blas = allocator->create_acceleration_structure(
                vk::AccelerationStructureTypeKHR::eBottomLevel, size_info,
                fmt::format("Scene::blas[{}]", group_id));
        }

        as_builder.queue_build(blas_geometry.geometries, blas_geometry.ranges, group.blas,
                               size_info, group.blas_build_flags);
        group.blas_dirty = false;
        group.blas_last_built_frame = current_frame;
        group.blas_last_updated_frame = current_frame;
        frame_stats.blas_builds++;
        if (static_geometry)
            frame_stats.blas_builds_static++;
        else
            frame_stats.blas_builds_dynamic++;
    }

    // Partial rebuild for update-eligible groups (morphed, fixed topology, existing BLAS)
    if (!update_eligible.empty()) {
        const uint32_t eligible_count = static_cast<uint32_t>(update_eligible.size());
        const uint32_t rebuild_count =
            static_cast<uint32_t>(std::ceil(eligible_count * blas_rebuild_fraction));

        if (rebuild_count < eligible_count) {
            std::sort(update_eligible.begin(), update_eligible.end(),
                      [&](MeshGroupID a, MeshGroupID b) {
                          return mesh_groups[a].blas_last_built_frame <
                                 mesh_groups[b].blas_last_built_frame;
                      });
        }

        for (uint32_t i = 0; i < eligible_count; i++) {
            const MeshGroupID group_id = update_eligible[i];
            MeshGroup& group = mesh_groups[group_id];
            auto& blas_geometry = blas_geometries[group_id];

            if (!group.cached_blas_size_info) {
                group.cached_blas_size_info = as_builder.get_size_info(
                    blas_geometry.geometries, blas_geometry.ranges, group.blas_build_flags);
            }
            const auto& size_info = *group.cached_blas_size_info;

            if (i < rebuild_count) {
                if (group.blas->get_size() < size_info.accelerationStructureSize) {
                    group.blas = allocator->create_acceleration_structure(
                        vk::AccelerationStructureTypeKHR::eBottomLevel, size_info,
                        fmt::format("Scene::blas[{}]", group_id));
                }
                as_builder.queue_build(blas_geometry.geometries, blas_geometry.ranges, group.blas,
                                       size_info, group.blas_build_flags);
                group.blas_last_built_frame = current_frame;
                frame_stats.blas_builds++;
                frame_stats.blas_builds_dynamic++;
            } else {
                as_builder.queue_update(blas_geometry.geometries, blas_geometry.ranges, group.blas,
                                        size_info, group.blas_build_flags);
                frame_stats.blas_updates++;
            }
            group.blas_dirty = false;
            group.blas_last_updated_frame = current_frame;
        }
    }

    as_builder.get_cmds_blas(cmd, as_scratch_buffer);

    if (did_build_static && as_scratch_buffer &&
        as_scratch_buffer->get_size() > MIN_BUFFER_CAPACITY) {
        as_scratch_buffer.reset();
    }
}

void Scene::build_tlas(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::build_tlas");

    tlas_instances.clear();
    // we set this such that GeometryData index is InstanceID + GeometryIndex.
    uint32_t instance_id = 0;

    for (uint32_t group_id = 0; group_id < mesh_groups.size(); group_id++) {
        // iterate in the same way as upload_geometry_data_and_transforms!

        const auto& group = mesh_groups[group_id];
        assert(group.blas);

        for (NodeID node_id : group.get_instances(mesh_infos)) {
            vk::AccelerationStructureInstanceKHR tlas_instance{};

            tlas_instance.instanceCustomIndex = instance_id;
            tlas_instance.mask = 0xFF;
            tlas_instance.accelerationStructureReference =
                group.blas->get_acceleration_structure_device_address();

            if (group.is_pretranformed(mesh_infos, pretransform_animated)) {
                // Use identity
                tlas_instance.transform.matrix[0][0] = 1.f;
                tlas_instance.transform.matrix[1][1] = 1.f;
                tlas_instance.transform.matrix[2][2] = 1.f;
            } else {
                const auto& t = get_global_transform(node_id);
                for (int row = 0; row < 3; row++)
                    for (int col = 0; col < 4; col++)
                        tlas_instance.transform.matrix[row][col] = t[row][col];
            }

            vk::GeometryInstanceFlagsKHR geometry_instance_flags{};
            if (group.flags & MeshFlags::FrontCounterClockwise) {
                geometry_instance_flags |=
                    vk::GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise;
            }
            if (group.flags & MeshFlags::TwoSided) {
                geometry_instance_flags |=
                    vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable;
            }
            if (group.all_opaque) {
                geometry_instance_flags |= vk::GeometryInstanceFlagBitsKHR::eForceOpaque;
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
        cmd->keep_until_pool_reset(std::move(tlas_instances_buffer));
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
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::update");
    current_frame = frame;
    frame_stats = {};

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "on_update");
        on_update(cmd, time, time_diff, frame);
    }

    assert(!cameras.empty() &&
           "the scene implementation must ensure that there is at least one camera");

    frame_stats.buffers_released = static_cast<uint32_t>(pending_buffer_releases.size());
    for (auto& buffer : pending_buffer_releases) {
        cmd->keep_until_pool_reset(std::move(buffer));
    }
    pending_buffer_releases.clear();

    for (auto& blas : pending_blas_releases) {
        cmd->keep_until_pool_reset(std::move(blas));
    }
    pending_blas_releases.clear();

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "material_system::update");
        material_system->update(cmd);
    }

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

    if (!vertex_jobs.empty() || !prev_vertex_jobs.empty()) {
        stage_transform_jobs(cmd);

        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        });

        dispatch_batched_transforms(cmd);
    }

    // Advance prev transforms after upload_meshes (prev-vertex jobs read them).
    for (const NodeID& nid : node_ids) {
        assert(scene_graph[nid]);
        SceneNode& n = *scene_graph[nid];
        n.prev_global_transform = n.global_transform;
        n.prev_global_inverse_transposed = n.global_inverse_transposed;
    }

    // changes at regroup, or updated mesh (buffer address), or if material ids change -> run every
    // frame.
    upload_geometry_data(cmd);

    frame_stats.geometry_data_bytes = geometries.size() * sizeof(GeometryData);
    frame_stats.transform_data_bytes = instance_transforms.size() * sizeof(float4x4) * 4;

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    });

    if (true /*build_as*/) {
        build_blas(cmd);
        // TODO: compact the static BLASes -> reduce memory bandwidth and increase performance.

        tlas_dirty |= needs_regroup || transforms_changed;

        if (tlas_dirty || !tlas) {
            build_tlas(cmd);
            tlas_dirty = false;
            frame_stats.tlas_rebuilt = true;
            frame_stats.tlas_instance_count = static_cast<uint32_t>(tlas_instances.size());
            frame_stats.tlas_instance_data_bytes =
                tlas_instances.size() * sizeof(vk::AccelerationStructureInstanceKHR);
        }

        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer |
                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eTransferWrite |
                vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
        });
    }

    for (const NodeID& nid : node_ids) {
        scene_graph[nid]->transform_dirty = false;
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
