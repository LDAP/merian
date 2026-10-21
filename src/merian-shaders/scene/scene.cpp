#include "merian-shaders/scene/scene.hpp"

#include "merian/io/image_io.hpp"
#include "merian/shader/entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/shader/spriv_reflect.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/math.hpp"

#include "transform_vertex.slang.spv.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/format.h>
#include <unordered_map>
#include <utility>

namespace merian {

// --- Constructor ---

Scene::Scene(const ShaderCompileContextHandle& compile_context,
             const ContextHandle& context,
             const ResourceAllocatorHandle& allocator,
             const MaterialSystemHandle& material_system)
    : compile_context(compile_context), context(context), allocator(allocator),
      as_builder(context, allocator), as_supported(context->get_device()
                                                       ->get_enabled_features()
                                                       .get_acceleration_structure_features_khr()
                                                       .accelerationStructure == VK_TRUE),
      material_system(material_system), lights(compile_context, context, allocator) {

    composition = SlangComposition::create();
    composition->add_composition(material_system->get_composition());

    // Workaround for https://github.com/shader-slang/slang/pull/10769.
    composition->add_module_from_string(
        "scene_as_workaround",
        as_supported ? "module scene_as_workaround;\n"
                       "import merian_shaders.scene.acceleration_structure;\n"
                       "namespace merian { public typealias SceneAccelerationStructure = "
                       "HWAccelerationStructure; }"
                     : "module scene_as_workaround;\n"
                       "import merian_shaders.scene.acceleration_structure;\n"
                       "namespace merian { public typealias SceneAccelerationStructure = "
                       "NullAccelerationStructure; }");
    set_env(std::make_shared<EmptyEnvMap>());
    prev_env_map = env_map->clone();
    compose_prev_env();

    if (as_supported) {
        micromaps = std::make_unique<Micromaps>(compile_context, context, allocator);
    }

    // register the hint modules up front so a runtime toggle replaces them instead of adding them
    set_enable_thin_lens(false);
    set_exterior_volume(std::make_shared<VacuumVolume>());

    composition->add_module_from_path("merian-shaders/scene/scene.slang");

    layout_program = SlangProgram::create(compile_context, composition);
    shader_object = Versioned<ShaderObject>([this] { return build_shader_object(); });
    shader_object.depends_on(layout_program);
}

DeviceSupportInfo Scene::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const SpirvReflect transform(merian_transform_vertex_slang_spv(),
                                 merian_transform_vertex_slang_spv_size());
    return DeviceSupportInfo::check(query_info, {},
                                    {"accelerationStructure", "micromap",
                                     "storageBuffer16BitAccess", "storageBuffer8BitAccess",
                                     "uniformAndStorageBuffer8BitAccess"}) &
           transform.query_device_support(query_info);
}

SlangCompositionHandle
Scene::query_device_support_composition(const DeviceSupportQueryInfo& query_info) {
    const bool as_supported = query_info.physical_device->get_supported_features()
                                  .get_acceleration_structure_features_khr()
                                  .accelerationStructure == VK_TRUE;

    const auto composition = SlangComposition::create();
    composition->add_composition(MaterialSystem::query_device_support_composition());
    composition->add_module_from_string(
        "scene_as_workaround",
        fmt::format("module scene_as_workaround;\n"
                    "import merian_shaders.scene.acceleration_structure;\n"
                    "namespace merian {{ public typealias SceneAccelerationStructure = {}; }}",
                    as_supported ? "HWAccelerationStructure" : "NullAccelerationStructure"));
    composition->add_module_from_string(
        "scene_env_light",
        "namespace merian { export static const bool merian_scene_has_env_light = false; }");
    composition->add_module_from_string(
        "scene_env_map_workaround",
        "module scene_env_map_workaround;\n"
        "import merian_shaders.scene.environment_map;\n"
        "namespace merian { public typealias SceneEnvMap = merian::EmptyEnv; }");
    composition->add_module_from_string(
        "scene_prev_env_map_workaround",
        "module scene_prev_env_map_workaround;\n"
        "import merian_shaders.scene.environment_map;\n"
        "namespace merian { public typealias ScenePrevEnvMap = merian::EmptyEnv; }");
    composition->add_module_from_string(
        "scene_volume_workaround",
        "module scene_volume_workaround;\n"
        "import merian_shaders.shading.homogeneous_volume;\n"
        "namespace merian { public typealias SceneHomogeneousVolume = merian::Vacuum; }");
    composition->add_module_from_path("merian-shaders/scene/scene.slang");
    return composition;
}

void Scene::set_env(EnvMapHandle env) {
    const bool needs_composition_update =
        !env_map || env_map->get_type_name() != env->get_type_name();
    env_map = std::move(env);
    if (!needs_composition_update) {
        return;
    }

    composition->add_module_from_string(
        "scene_env_light",
        fmt::format("namespace merian {{ export static const bool merian_scene_has_env_light = "
                    "{}; }}",
                    env_map->is_emissive() ? "true" : "false"));
    composition->add_module(env_map->get_slang_module());
    composition->add_module_from_string(
        "scene_env_map_workaround",
        fmt::format("module scene_env_map_workaround;\n"
                    "import {};\n"
                    "namespace merian {{ public typealias SceneEnvMap = {}; }}",
                    env_map->get_slang_module().get_import_spelling(), env_map->get_type_name()));
}

void Scene::compose_prev_env() {
    if (prev_env_map->get_type_name() == prev_env_type_name) {
        return;
    }
    prev_env_type_name = prev_env_map->get_type_name();
    composition->add_module(prev_env_map->get_slang_module());
    composition->add_module_from_string(
        "scene_prev_env_map_workaround",
        fmt::format("module scene_prev_env_map_workaround;\n"
                    "import {};\n"
                    "namespace merian {{ public typealias ScenePrevEnvMap = {}; }}",
                    prev_env_map->get_slang_module().get_import_spelling(), prev_env_type_name));
}

void Scene::set_enable_thin_lens(const bool enable) {
    thin_lens_enabled = enable;
    composition->add_module_from_string("scene_thin_lens",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_hint_enable_thin_lens = {}; }}",
                                                    enable ? "true" : "false"));
}

void Scene::set_exterior_volume(const HomogeneousVolumeHandle& volume) {
    const bool needs_composition_update =
        !exterior_volume || exterior_volume->get_type_name() != volume->get_type_name();
    exterior_volume = volume;
    if (!needs_composition_update) {
        return;
    }

    composition->add_module(exterior_volume->get_slang_module());
    composition->add_module_from_string(
        "scene_volume_workaround",
        fmt::format("module scene_volume_workaround;\n"
                    "import {};\n"
                    "namespace merian {{ public typealias SceneHomogeneousVolume = {}; }}",
                    exterior_volume->get_slang_module().get_import_spelling(),
                    exterior_volume->get_type_name()));
}

ShaderObjectHandle Scene::build_shader_object() const {
    SPDLOG_DEBUG("recreate shader object");

    const ShaderObjectHandle object =
        layout_program->create_shader_object_for_type(context, "merian::Scene", allocator);

    auto c = object->get_cursor();

    c["material_system"] = material_system;

    c["geometries"] = geometries_buffer ? geometries_buffer : allocator->get_dummy_buffer();
    c["prev_geometry_remap"] =
        prev_geometry_remap_buffer ? prev_geometry_remap_buffer : allocator->get_dummy_buffer();

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

    lights.write_to(c["nee"]);
    lights.write_prev_to(c["prev_nee"]);
    return object;
}

// --- Scene building ---

Scene::MeshID Scene::add_mesh(MeshHandle mesh) {
    assert(mesh);
    const MeshID id = mesh_ids.acquire();
    if (id >= mesh_infos.size()) {
        mesh_infos.resize(mesh_ids.size());
    }
    mesh_infos[id].mesh = std::move(mesh);
    mesh_infos[id].serial = next_serial++;

    return id;
}

Scene::NodeID Scene::add_node(Node node) {
    const NodeID id = node_ids.acquire();

    if (id >= scene_graph.size()) {
        scene_graph.resize(node_ids.size());
    }

    if (node.parent != NODE_ID_INVALID) {
        assert(node_ids.is_used(node.parent));
        assert(scene_graph[node.parent]);
        scene_graph[node.parent]->children.push_back(id);
    }

    node.meshes.clear();
    node.serial = next_serial++;
    scene_graph[id] = std::move(node);
    return id;
}

void Scene::invalidate_node(Node& node) {
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

    Node& node = *scene_graph[node_id];
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

    // vertices may still be pretransformed for an earlier instance
    if (info.instances.size() <= 1 && (!info.is_dynamic() || pretransform_animated)) {
        info.mesh->vertices_dirty = true;
    }
    info.instances.insert(node_id);
    scene_graph[node_id]->meshes.insert(mesh_id);
    if (scene_graph[node_id]->is_animated)
        info.animated_instance_count++;
    needs_regroup = true;
}

void Scene::remove_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_ids.is_used(mesh_id));
    assert(node_ids.is_used(node_id));

    MeshInfo& info = mesh_infos[mesh_id];

    if (info.instances.erase(node_id) > 0) {
        scene_graph[node_id]->meshes.erase(mesh_id);
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
        retire(std::move(group.blas));
        retire(std::move(group.spare_blas));
    }

    MeshInfo& info = mesh_infos[mesh_id];
    for (const NodeID node_id : info.instances) {
        scene_graph[node_id]->meshes.erase(mesh_id);
    }

    retire(std::move(info.vertex_buffer.suballoc));
    retire(std::move(info.prev_vertex_buffer.suballoc));
    retire(std::move(info.index_buffer.suballoc));
    info = MeshInfo{};

    mesh_ids.release(mesh_id);
    mesh_infos.resize(mesh_ids.size());

    needs_regroup = true;
}

void Scene::remove_node(const NodeID node_id) {
    assert(node_ids.is_used(node_id));
    Node& node = *scene_graph[node_id];

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

    for (const MeshID mesh_id : std::exchange(node.meshes, {})) {
        remove_mesh_instance(mesh_id, node_id);
    }

    needs_regroup = true;

    scene_graph[node_id].reset();
    node_ids.release(node_id);
    scene_graph.resize(node_ids.size());
}

Scene::CameraID Scene::add_camera(CameraHandle camera) {
    const CameraID id = cameras.size();
    cameras.push_back(std::move(camera));
    return id;
}

void Scene::retire(ConstObjectHandle object) {
    if (object) {
        retired.push_back(std::move(object));
    }
}

void Scene::clear_geometry() {
    // remove_node recurses into children, so a snapshot of root NodeIDs is enough.
    std::vector<NodeID> roots;
    for (const NodeID id : node_ids) {
        assert(scene_graph[id]);
        if (scene_graph[id]->parent == NODE_ID_INVALID) {
            roots.push_back(id);
        }
    }
    for (const NodeID id : roots) {
        if (node_ids.is_used(id)) {
            remove_node(id);
        }
    }

    // remove_mesh mutates mesh_ids; snapshot first.
    std::vector<MeshID> meshes;
    meshes.reserve(mesh_ids.count());
    for (const MeshID id : mesh_ids) {
        meshes.push_back(id);
    }
    for (const MeshID id : meshes) {
        remove_mesh(id);
    }

    cameras.clear();
    active_camera = 0;
    aabb.reset();
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

std::optional<vk::Extent3D> Scene::get_resolution() const {
    if (cameras.empty()) {
        return std::nullopt;
    }
    return get_active_camera()->get_resolution();
}

void Scene::set_active_camera(const uint32_t index) {
    assert(index < cameras.size());
    active_camera = index;
}

// --- Batched GPU transform ---

namespace {

constexpr uint32_t TRANSFORM_LOCAL_SIZE_X = 256;

} // namespace

void Scene::ensure_transform_pipelines() {
    if (transform_vertex_pipeline)
        return;

    transform_descriptor_layout =
        DescriptorSetLayoutBuilder()
            .add_binding_storage_buffer(1, vk::ShaderStageFlagBits::eCompute)
            .build_push_descriptor_layout(context);

    auto vertex_module = EntryPoint::create(context, merian_transform_vertex_slang_spv(),
                                            merian_transform_vertex_slang_spv_size(), "main",
                                            vk::ShaderStageFlagBits::eCompute);

    const auto layout = PipelineLayoutBuilder(context)
                            .add_descriptor_set_layout(transform_descriptor_layout)
                            .build_pipeline_layout();

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(TRANSFORM_LOCAL_SIZE_X);
    const SpecializationInfoHandle spec = spec_builder.build();

    transform_vertex_pipeline = ComputePipeline::create(layout, vertex_module, spec);
}

void Scene::stage_transform_jobs(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::stage_transform_jobs");
    const auto staging = allocator->get_staging();
    const auto job_buf_usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;

    const vk::DeviceSize size = vertex_jobs.size() * sizeof(TransformVertexJob);
    if (!transform_vertex_job_buf || transform_vertex_job_buf->get_size() < size) {
        retire(std::move(transform_vertex_job_buf));
        transform_vertex_job_buf = allocator->create_buffer(
            size, job_buf_usage, MemoryMappingType::NONE, "Scene::transform_vertex_jobs");
        frame_stats.buffers_allocated++;
    }
    staging->cmd_to_device(cmd, transform_vertex_job_buf, vertex_jobs.data(), 0, size);
    frame_stats.gpu_transform_buffer_bytes += size;
}

void Scene::dispatch_batched_transforms(const CommandBufferHandle& cmd) {
    ensure_transform_pipelines();
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::transform_vertices");

    const uint32_t wgx =
        (max_vertex_count_in_jobs + TRANSFORM_LOCAL_SIZE_X - 1) / TRANSFORM_LOCAL_SIZE_X;
    cmd->bind(transform_vertex_pipeline);
    cmd->push_descriptor_set(transform_vertex_pipeline,
                             transform_vertex_job_buf->get_descriptor_info());
    cmd->dispatch(wgx, static_cast<uint32_t>(vertex_jobs.size()), 1);
}

void Scene::set_pretransform_animated(bool value) {
    if (pretransform_animated == value)
        return;

    pretransform_animated = value;

    // Vertex-buffer contents (pretransformed or not) flip for single-instance dynamic meshes.
    for (const MeshID mesh_id : mesh_ids) {
        MeshInfo& info = mesh_infos[mesh_id];
        if (info.is_dynamic() && info.instances.size() == 1) {
            info.mesh->vertices_dirty = true;
        }
    }

    needs_regroup = true;
}

void Scene::properties_node(Properties& props, const NodeID node_id) {
    if (!props.is_ui())
        return;
    if (!node_ids.is_used(node_id) || node_id >= scene_graph.size() || !scene_graph[node_id]) {
        props.output_text("<node {} not available>", node_id);
        return;
    }
    const Node& node = *scene_graph[node_id];
    props.output_text("{}", node);

    for (uint32_t i = 0; i < node.children.size(); i++) {
        const NodeID child_id = node.children[i];
        const Node& child = *scene_graph[child_id];
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
    Mesh& mesh = *info.mesh;
    const MeshGroupID group_id =
        mesh_id < mesh_to_group.size() ? mesh_to_group[mesh_id] : MESH_GROUP_ID_INVALID;

    props.output_text("{}", mesh);

    // Names sourced from format_as(MeshFlags) so this stays in sync with the enum.
    if (props.st_begin_child("flags", "flags")) {
        auto flag_bits = static_cast<uint32_t>(mesh.flags);
        for (uint32_t bit = 0; bit < 8; ++bit) {
            const std::string name = fmt::to_string(static_cast<MeshFlags>(1u << bit));
            if (name.empty())
                continue;
            bool on = (flag_bits & (1u << bit)) != 0u;
            if (props.config_bool(name, on)) {
                flag_bits = (flag_bits & ~(1u << bit)) | (on ? (1u << bit) : 0u);
                mesh.flags = static_cast<MeshFlags>(flag_bits);
                needs_regroup = true;
            }
        }
        props.st_end_child();
    }

    // instance_mask is part of the group split key, so any change forces a regroup.
    if (props.st_begin_child("instance mask", "instance mask")) {
        for (uint32_t bit = 0; bit < 8; ++bit) {
            bool on = (mesh.instance_mask & (1u << bit)) != 0u;
            if (props.config_bool(std::to_string(bit), on)) {
                mesh.instance_mask = static_cast<uint8_t>((mesh.instance_mask & ~(1u << bit)) |
                                                          (on ? (1u << bit) : 0u));
                needs_regroup = true;
            }
            if ((bit & 3u) != 3u)
                props.st_no_space();
        }
        props.st_end_child();
    }

    if (group_id != MESH_GROUP_ID_INVALID &&
        props.st_begin_child("group", fmt::format("group {}", group_id))) {
        properties_mesh_group(props, group_id);
        props.st_end_child();
    }

    if (props.st_begin_child("instances", fmt::format("instances ({})", info.instances.size()))) {
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
    const std::size_t instances = group.meshes.empty() ? 0 : group.get_instances(mesh_infos).size();
    const bool pretransformed =
        !group.meshes.empty() && group.is_pretransformed(mesh_infos, pretransform_animated);

    std::size_t group_vertices = 0;
    std::size_t group_triangles = 0;
    for (const MeshID mesh_id : group.meshes) {
        group_vertices += mesh_infos[mesh_id].mesh->get_vertex_count();
        group_triangles += mesh_infos[mesh_id].mesh->get_primitive_count();
    }
    props.output_text(
        "split flags: {}\nhas animated node: {}\nhas morphed mesh: {}\nhas variable topology: "
        "{}\nall opaque: {}\npretransformed: {}\ninstances: {}\nvertices: {}\ntriangles: "
        "{}\nblas: {}\nblas build flags: {}\nblas dirty: {}\nblas last built update: {}\nblas "
        "last updated update: {}",
        group.flags, group.has_animated_node, group.has_morphed_mesh,
        group.has_variable_topology_mesh, group.all_opaque, pretransformed, instances,
        group_vertices, group_triangles,
        group.blas ? format_size(group.blas->get_size()) : "<none>",
        vk::to_string(group.blas_build_flags), group.blas_dirty, group.blas_last_built_update,
        group.blas_last_updated_update);

    if (props.st_begin_child("members", fmt::format("members ({})", group.meshes.size()))) {
        for (const MeshID mesh_id : group.meshes) {
            const std::string mesh_name =
                mesh_infos[mesh_id].mesh ? mesh_infos[mesh_id].mesh->name : std::string{"<dead>"};
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
        props.config_bool("store cameras", store_cameras,
                          "persist the active camera in stored graphs");
        props.config_uint("active", active_camera, "", 0u, static_cast<uint32_t>(cameras.size()));
        if (props.is_ui() || store_cameras) {
            props.st_separate("Active Camera");
            get_active_camera()->properties(props);
        }
        if (props.is_ui() && aabb.is_valid() &&
            props.config_bool("fit aabb", "Rotates and moves the camera to fit the scene AABB.")) {
            get_active_camera()->look_at_bounding_box(aabb);
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
        // single-mesh groups: surface the mesh inline so the explorer stays navigable.
        std::string label;
        if (group.meshes.size() == 1) {
            const MeshID only = *group.meshes.begin();
            const std::string mesh_name =
                mesh_infos[only].mesh ? mesh_infos[only].mesh->name : std::string{"<dead>"};
            label = fmt::format("{:04}: {} (1 mesh, {:04}: {})", group_id, group.flags, only,
                                mesh_name);
        } else {
            label =
                fmt::format("{:04}: {} ({} meshes)", group_id, group.flags, group.meshes.size());
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
        const Node& node = *scene_graph[id];
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
    if (props.st_begin_child("volume", "Exterior Volume")) {
        properties_volume(props);
        props.st_end_child();
    }
    if (!props.is_ui())
        return;
    if (props.st_begin_child("meshes", fmt::format("Meshes ({})", mesh_ids.count()))) {
        properties_meshes(props);
        props.st_end_child();
    }
    if (props.st_begin_child("mesh groups", fmt::format("Mesh Groups ({})", mesh_groups.size()))) {
        properties_mesh_groups(props);
        props.st_end_child();
    }
    if (props.st_begin_child("graph", fmt::format("Graph ({} nodes)", node_ids.count()))) {
        properties_graph(props);
        props.st_end_child();
    }
    if (props.st_begin_child("material system", "Material System")) {
        material_system->properties(props);
        props.st_end_child();
    }
    if (props.st_begin_child("texture manager", "Texture Manager")) {
        get_texture_manager()->properties(props);
        props.st_end_child();
    }
}

void Scene::properties_settings(Properties& props) {
    bool pretransform = get_pretransform_animated();
    if (props.config_bool("pretransform animated", pretransform)) {
        set_pretransform_animated(pretransform);
    }
    props.config_percent("blas rebuild fraction", blas_rebuild_fraction);

    props.st_separate("Micromaps");
    if (micromaps && Micromaps::is_opacity_supported(context)) {
        props.config_bool("opacity micromaps", opacity_micromaps_enabled,
                          "Settle alpha-tested hits during traversal instead of invoking the alpha "
                          "test");
    } else {
        props.output_text("opacity micromaps: not supported by the device");
    }
    if (micromaps) {
        props.config_bool("emission micromaps", emission_micromaps_enabled,
                          "Draw light samples on an emissive triangle only where its emission "
                          "texture emits");
        micromaps->properties(props);
    }

    props.st_separate("Material System");
    float alpha_threshold = material_system->get_alpha_test_threshold();
    if (props.config_float("alpha test threshold", alpha_threshold, "", 0.01F)) {
        material_system->set_alpha_test_threshold(alpha_threshold);
    }
    bool clamp_normals = material_system->get_clamp_normals();
    if (props.config_bool("clamp normals", clamp_normals,
                          "Hint the materials to clamp their normals to prevent artifacts when "
                          "using normal maps")) {
        material_system->set_clamp_normals(clamp_normals);
    }
}

void Scene::process_pending_env_load(const CommandBufferHandle& cmd) {
    if (!pending_env_load) {
        return;
    }

    const auto load_texture = [&](const std::filesystem::path& path,
                                  const std::string& name) -> TextureHandle {
        ImageInfo info;
        const ImageFormat fmt = image_format_from_extension(path);
        const bool is_hdr = fmt == ImageFormat::HDR || fmt == ImageFormat::PFM;
        if (is_hdr) {
            const BlobHandle blob = image_load_f32(path, info);
            return allocator->create_texture_from_rgba32f(
                cmd, blob->get_data<float>(), info.width, info.height,
                vk::SamplerAddressMode::eRepeat, vk::Filter::eLinear, vk::Filter::eLinear, name);
        }
        const BlobHandle blob = image_load_u8(path, info);
        return allocator->create_texture_from_rgba8(
            cmd, blob->get_data<uint32_t>(), info.width, info.height,
            vk::SamplerAddressMode::eRepeat, vk::Filter::eLinear, vk::Filter::eLinear, true, name);
    };

    try {
        if (pending_env_load->kind == 1) {
            const TextureHandle tex = load_texture(pending_env_load->latlong_path, "env_latlong");
            cmd->barrier(tex->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
            if (auto e = std::dynamic_pointer_cast<LatLongEnvMap>(env_map)) {
                e->set_texture(tex);
            }
        } else if (pending_env_load->kind == 3) {
            const TextureHandle tex = load_texture(pending_env_load->latlong_path, "env_oct");
            cmd->barrier(tex->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
            if (auto e = std::dynamic_pointer_cast<EqualAreaOctEnvMap>(env_map)) {
                e->set_texture(tex);
            }
        } else if (pending_env_load->kind == 2) {
            std::array<TextureHandle, 6> faces;
            for (int i = 0; i < 6; ++i) {
                faces[i] =
                    load_texture(pending_env_load->cubemap_paths[i], fmt::format("env_cube_{}", i));
                cmd->barrier(
                    faces[i]->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
            }
            if (auto e = std::dynamic_pointer_cast<CubeMapEnvMap>(env_map)) {
                e->set_faces(faces);
            }
        }
    } catch (const std::exception& e) {
        SPDLOG_ERROR("env map load failed: {}", e.what());
    }
    pending_env_load.reset();
}

void Scene::properties_env(Properties& props) {
    static const std::vector<std::string> kinds = {"Empty", "LatLong", "CubeMap", "EqualAreaOct"};
    const bool selection_changed = props.config_options("type", env_ui.selection, kinds);

    bool paths_changed = false;
    if (env_ui.selection == 1 || env_ui.selection == 3) {
        paths_changed |= props.config_path("path", env_ui.latlong_path);
    } else if (env_ui.selection == 2) {
        static constexpr std::array<const char*, 6> face_labels = {"rt (+X)", "bk (+Y)", "lf (-X)",
                                                                   "ft (-Y)", "up (+Z)", "dn (-Z)"};
        for (int i = 0; i < 6; ++i) {
            paths_changed |= props.config_path(face_labels[i], env_ui.cubemap_paths[i]);
        }
    }

    if (selection_changed) {
        const float old_yaw = env_map ? env_map->get_yaw_rad() : 0.f;
        const float old_pitch = env_map ? env_map->get_pitch_rad() : 0.f;

        const TextureHandle dummy = allocator->get_dummy_texture();
        EnvMapHandle next;
        if (env_ui.selection == 1) {
            next = std::make_shared<LatLongEnvMap>(dummy);
        } else if (env_ui.selection == 2) {
            next = std::make_shared<CubeMapEnvMap>(
                std::array<TextureHandle, 6>{dummy, dummy, dummy, dummy, dummy, dummy});
        } else if (env_ui.selection == 3) {
            next = std::make_shared<EqualAreaOctEnvMap>(dummy);
        } else {
            next = std::make_shared<EmptyEnvMap>();
        }
        next->set_base_transform(rotation_from_to(get_up(), float3(0, 1, 0)));
        next->set_yaw_pitch(old_yaw, old_pitch);
        set_env(next);
    }

    if ((selection_changed || paths_changed) && env_ui.selection != 0) {
        pending_env_load =
            PendingEnvLoad{env_ui.selection, env_ui.latlong_path, env_ui.cubemap_paths};
    }

    if (env_map) {
        env_map->properties(props);
    }
}

void Scene::properties_lights(Properties& props) {
    lights.properties(props);
}

void Scene::properties_volume(Properties& props) {
    static const std::vector<std::string> kinds = {"Vacuum", "Fog"};
    // the volume can also be set by the scene implementation, so the current type is the state
    int selection = exterior_volume->get_type_name() == "merian::Fog" ? 1 : 0;

    if (props.config_options("type", selection, kinds)) {
        if (selection == 1) {
            const auto fog = std::make_shared<FogVolume>();
            // optical depth of about one across the scene, so the fog shows at any scene scale
            const float extent = aabb.is_valid() ? length(aabb.get_max() - aabb.get_min()) : 0.f;
            fog->mu_t = float3(extent > 0.f ? 1.f / extent : 1.f);
            set_exterior_volume(fog);
        } else {
            set_exterior_volume(std::make_shared<VacuumVolume>());
        }
    }

    exterior_volume->properties(props);
}

void Scene::properties_statistics(Properties& props) {
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
        "retired:     {}",
        node_ids.count(), mesh_ids.count(), dynamic_meshes, morphed_meshes,
        animated_instance_meshes, variable_topology_meshes, total_instances, animated_instances,
        mesh_groups.size(), total_vertices, total_triangles, material_system->get_material_count(),
        get_texture_manager()->get_texture_count(), tlas_instances.size(), needs_regroup,
        transforms_changed, pretransform_animated, retired.size());

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
        "xfm job buffers: {}\n"
        "blas:            {} ops (builds_static: {}, builds_dynamic: {}, updates: {})\n"
        "tlas:            {} ({} instances)\n"
        "buffers:         allocated: {}, retired objects: {}\n"
        "gpu data:        geometry {}, transforms {}, tlas instances {}",
        frame_stats.meshes_uploaded(), frame_stats.meshes_uploaded_device_local,
        frame_stats.meshes_uploaded_device_staged, frame_stats.meshes_uploaded_host_packed,
        frame_stats.meshes_uploaded_host_unpacked, format_size(frame_stats.upload_bytes),
        frame_stats.vertices_uploaded, frame_stats.indices_uploaded,
        frame_stats.gpu_vertex_transforms, frame_stats.gpu_vertex_transform_vertices,
        format_size(frame_stats.gpu_transform_buffer_bytes),
        frame_stats.blas_builds + frame_stats.blas_updates, frame_stats.blas_builds_static,
        frame_stats.blas_builds_dynamic, frame_stats.blas_updates,
        frame_stats.tlas_rebuilt ? "rebuilt" : "unchanged", frame_stats.tlas_instance_count,
        frame_stats.buffers_allocated, frame_stats.objects_retired,
        format_size(frame_stats.geometry_data_bytes), format_size(frame_stats.transform_data_bytes),
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
    block_text("shared_ib", shared_ib_suballoc, shared_ib_capacity);
    props.output_text("{:<14} {} staged this frame",
                      "upload_staging:", format_size(frame_stats.upload_bytes));
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
    if (props.st_begin_child("env", "Environment Map")) {
        properties_env(props);
        props.st_end_child();
    }
    if (props.st_begin_child("lights", "Lights")) {
        properties_lights(props);
        props.st_end_child();
    }
    if (props.st_begin_child("stats", "Statistics")) {
        properties_statistics(props);
        props.st_end_child();
    }
}

// --- Scene update ---

void Scene::compute_mesh_groups() {
    MERIAN_PROFILE_SCOPE("Scene::compute_mesh_groups");

    auto prev_mesh_groups = std::move(mesh_groups);
    auto prev_mesh_to_group = std::move(mesh_to_group);
    mesh_groups.clear();
    mesh_to_group.assign(mesh_ids.size(), MESH_GROUP_ID_INVALID);

    // allow to access prev_mesh_to_group with any (new) MeshID.
    prev_mesh_to_group.resize(mesh_ids.size(), MESH_GROUP_ID_INVALID);
    mesh_groups.reserve(std::min<std::size_t>(mesh_ids.count(), prev_mesh_groups.size()));

    // 1. Group meshes according to the grouping logic in scene.hpp. The split key
    //    holds the TLAS-instance flags (GROUP_SPLIT_MASK) plus instance_mask.
    const auto split_key = [](MeshFlags f, uint8_t mask) -> uint32_t {
        return (static_cast<uint32_t>(f) & static_cast<uint32_t>(GROUP_SPLIT_MASK)) |
               (static_cast<uint32_t>(mask) << 16);
    };

    // static (non-morphed, non-animated): pretransformed, BLAS built once
    std::unordered_map<uint32_t, MeshGroupID> groups_static;
    // dynamic with pretransform_animated: pretransformed, BLAS rebuilt as needed
    std::unordered_map<uint32_t, MeshGroupID> groups_dynamic_pretransformed;
    // dynamic, pretransform OFF: not pretransformed, per-node
    std::unordered_map<uint64_t, MeshGroupID> groups_dynamic_per_node;
    // instanced
    std::unordered_map<SmallSet<NodeID, 1>, std::unordered_map<uint32_t, MeshGroupID>>
        groups_instanced;

    groups_static.reserve(32);
    groups_dynamic_per_node.reserve(node_ids.count() * 2);

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

        const uint32_t key = split_key(mesh.flags, mesh.instance_mask);
        MeshGroupID group_id;

        if (info.instances.size() > 1) {
            group_id = get_or_create(groups_instanced[info.instances], key);
        } else {
            assert(info.instances.size() == 1);
            const NodeID node_id = *info.instances.begin();
            const bool is_dynamic = scene_graph[node_id]->is_animated || mesh.is_morphed();

            if (!is_dynamic) {
                group_id = get_or_create(groups_static, key);
            } else if (pretransform_animated) {
                group_id = get_or_create(groups_dynamic_pretransformed, key);
            } else {
                const uint64_t node_key = (static_cast<uint64_t>(key) << 32) | node_id;
                group_id = get_or_create(groups_dynamic_per_node, node_key);
            }
        }

        mesh_to_group[mesh_id] = group_id;
        MeshGroup& group = mesh_groups[group_id];
        group.flags = static_cast<MeshFlags>(static_cast<uint32_t>(GROUP_SPLIT_MASK) & key);
        group.instance_mask = mesh.instance_mask;
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

        // BLAS-space vertices change when the mesh is morphed or pretransform bakes
        // a moving node transform in; an animated node alone only moves the TLAS instance.
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

        // Checking the first mesh suffices — if it changed group, the BLAS can't be reused.
        MeshGroupID maybe_prev_group = prev_mesh_to_group[*group.meshes.begin()];
        if (maybe_prev_group != MESH_GROUP_ID_INVALID) {
            MeshGroup& prev_group = prev_mesh_groups[maybe_prev_group];
            if (prev_group.meshes == group.meshes &&
                prev_group.blas_build_flags == group.blas_build_flags) {
                group.blas = std::move(prev_group.blas);
                group.spare_blas = std::move(prev_group.spare_blas);
                group.cached_blas_size_info = prev_group.cached_blas_size_info;
            }
        }
    }

    for (auto& prev_group : prev_mesh_groups) {
        retire(std::move(prev_group.blas));
        retire(std::move(prev_group.spare_blas));
    }
}

void Scene::upload_transforms(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::upload_transforms");
    if (geometry.instance_transforms.empty()) {
        return;
    }

    const vk::DeviceSize transforms_size = geometry.instance_transforms.size() * sizeof(float4x4);
    if (!instance_transforms_buffer || instance_transforms_buffer->get_size() < transforms_size) {
        retire(std::move(instance_transforms_buffer));
        retire(std::move(inverse_transposed_instance_transforms_buffer));
        retire(std::move(prev_instance_transforms_buffer));
        retire(std::move(prev_inverse_transposed_instance_transforms_buffer));

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

        auto c = shader_object->get_cursor();
        c["instance_transforms"] = instance_transforms_buffer;
        c["inverse_transposed_instance_transforms"] = inverse_transposed_instance_transforms_buffer;
        c["prev_instance_transforms"] = prev_instance_transforms_buffer;
        c["prev_inverse_transposed_instance_transforms"] =
            prev_inverse_transposed_instance_transforms_buffer;
    }

    const auto staging = allocator->get_staging();
    staging->cmd_to_device(cmd, instance_transforms_buffer, geometry.instance_transforms);
    staging->cmd_to_device(cmd, inverse_transposed_instance_transforms_buffer,
                           geometry.inverse_transposed_instance_transforms);
    staging->cmd_to_device(cmd, prev_instance_transforms_buffer, geometry.prev_instance_transforms);
    staging->cmd_to_device(cmd, prev_inverse_transposed_instance_transforms_buffer,
                           geometry.prev_inverse_transposed_instance_transforms);
}

void Scene::upload_geometry_data(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::upload_geometry_data");
    geometry.geometries.clear();
    geometry.geometry_keys.clear();
    geometry.instance_indices.clear();
    geometry.instance_transforms.clear();
    geometry.inverse_transposed_instance_transforms.clear();
    geometry.prev_instance_transforms.clear();
    geometry.prev_inverse_transposed_instance_transforms.clear();
    emissive_geometries.clear();
    bool has_sky_portals = false;

    const float4x4 identity_transform = identity();
    // same (group, node) order as build_tlas
    uint32_t instance_index = 0;

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        const bool pretransform = group.is_pretransformed(mesh_infos, pretransform_animated);

        for (const NodeID node_id : group.get_instances(mesh_infos)) {
            Node& node = *scene_graph[node_id];
            const float4x4& global_transform = get_global_transform(node);
            const float4x4& global_inverse_transposed =
                get_global_inverse_transposed_transform(node);
            if (pretransform) {
                geometry.instance_transforms.emplace_back(identity_transform);
                geometry.inverse_transposed_instance_transforms.emplace_back(identity_transform);
                geometry.prev_instance_transforms.emplace_back(identity_transform);
                geometry.prev_inverse_transposed_instance_transforms.emplace_back(
                    identity_transform);
            } else {
                geometry.instance_transforms.emplace_back(global_transform);
                geometry.inverse_transposed_instance_transforms.emplace_back(
                    global_inverse_transposed);
                geometry.prev_instance_transforms.emplace_back(
                    starting_run ? global_transform
                                 : node.prev_global_transform.value_or(global_transform));
                geometry.prev_inverse_transposed_instance_transforms.emplace_back(
                    starting_run
                        ? global_inverse_transposed
                        : node.prev_global_inverse_transposed.value_or(global_inverse_transposed));
            }

            // an animated node passing through identity still has a previous transform
            const bool transform_is_identity =
                !node.is_animated && global_transform == identity_transform;

            for (const MeshID mesh_id : group.meshes) {
                MeshInfo& info = mesh_infos[mesh_id];
                Mesh& mesh = *info.mesh;
                assert(info.vertex_buffer && (!mesh.has_indices() || info.index_buffer));

                const uint64_t geometry_key =
                    (static_cast<uint64_t>(info.serial) << 32) | node.serial;

                has_sky_portals |= (mesh.flags & MeshFlags::UseEnvMap);
                const bool is_light = !(mesh.flags & MeshFlags::UseEnvMap) &&
                                      material_system->is_emissive(mesh.material_id);
                if (is_light) {
                    LightCollection::EmissiveGeometry& emissive =
                        emissive_geometries.emplace_back(LightCollection::EmissiveGeometry{
                            static_cast<GeometryID>(geometry.geometries.size()), instance_index,
                            mesh.get_primitive_count()});
                    if (micromaps) {
                        if (const auto emission = micromaps->get_emission(
                                mesh_id, info.serial, mesh.get_primitive_count())) {
                            emissive.emission_offsets = emission->offsets;
                            emissive.emission_records = emission->records;
                            emissive.emission_level = emission->level;
                        }
                    }
                }

                GeometryData gd;
                gd.material_id = mesh.material_id;
                gd.vertices = info.vertex_buffer.get_device_address();
                gd.prev_vertices = info.prev_vertex_buffer
                                       ? info.prev_vertex_buffer.get_device_address()
                                       : gd.vertices;
                gd.indices = info.index_buffer ? info.index_buffer.get_device_address()
                                               : vk::DeviceAddress{0};
                gd.flags = index_type_flag(mesh.index_type);
                if (pretransform || transform_is_identity) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::Pretransformed);
                }
                if (mesh.is_flip_facing()) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::FlipFacing);
                }
                if (mesh.flags & MeshFlags::HasTangents) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::HasTangents);
                }
                if (mesh.flags & MeshFlags::FlatShading) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::FlatShading);
                }
                if (mesh.flags & MeshFlags::UseEnvMap) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::UseEnvMap);
                }
                if (mesh.flags & MeshFlags::TwoSided) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::TwoSided);
                }
                if (!mesh.is_opaque() && material_system->has_alpha_texture(mesh.material_id)) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::AlphaTested);
                }
                if (is_light) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::Emissive);
                }

                geometry.geometries.emplace_back(gd);
                geometry.geometry_keys.push_back(geometry_key);
                geometry.instance_indices.push_back(instance_index);
            }
            instance_index++;
        }
    }

    lights.set_has_sky_portals(has_sky_portals);
    compute_prev_geometry_remap();
    upload_prev_geometry_remap(cmd);
    upload_transforms(cmd);

    const std::vector<GeometryData>& geometries = geometry.geometries;
    if (!geometries.empty()) {
        const vk::DeviceSize geometries_size = geometries.size() * sizeof(GeometryData);
        if (!geometries_buffer || geometries_buffer->get_size() < geometries_size) {
            retire(std::move(geometries_buffer));

            geometries_buffer = allocator->create_buffer(
                geometries_size,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                MemoryMappingType::NONE, "Scene::geometries");

            shader_object->get_cursor()["geometries"] = geometries_buffer;
        }
        allocator->get_staging()->cmd_to_device(cmd, geometries_buffer, geometries);
    }

    lights.set_geometries(emissive_geometries, static_cast<uint32_t>(geometries.size()),
                          prev_geometry_remap);

    for (const NodeID node_id : node_ids) {
        Node& node = *scene_graph[node_id];
        node.prev_global_transform = node.global_transform;
        node.prev_global_inverse_transposed = node.global_inverse_transposed;
    }
}

namespace {

bool continues(const GeometryData& geometry_data, const GeometryData& prev_geometry_data) {
    constexpr uint32_t frame_flags =
        GeometryDataFlags::Emissive | GeometryDataFlags::New | GeometryDataFlags::Retired;
    return geometry_data.material_id == prev_geometry_data.material_id &&
           (geometry_data.flags & ~frame_flags) == (prev_geometry_data.flags & ~frame_flags);
}

} // namespace

void Scene::compute_prev_geometry_remap() {
    prev_geometry_remap.clear();
    if (starting_run)
        return;

    const std::vector<uint64_t>& keys = geometry.geometry_keys;
    const std::vector<uint64_t>& prev_keys = prev_geometry.geometry_keys;
    const auto same_key = [&](const GeometryID geometry_id) {
        return geometry_id < keys.size() && geometry_id < prev_keys.size() &&
               keys[geometry_id] == prev_keys[geometry_id];
    };
    const auto continues_at = [&](const GeometryID geometry_id, const GeometryID prev_id) {
        return continues(geometry.geometries[geometry_id], prev_geometry.geometries[prev_id]);
    };

    bool unchanged = keys.size() == prev_keys.size();
    for (GeometryID geometry_id = 0; unchanged && geometry_id < keys.size(); geometry_id++)
        unchanged = same_key(geometry_id) && continues_at(geometry_id, geometry_id);
    if (unchanged)
        return;

    // keys are unique, so a key that moved lands where the key changed
    geometry_of_key.clear();
    for (GeometryID geometry_id = 0; geometry_id < keys.size(); geometry_id++)
        if (!same_key(geometry_id))
            geometry_of_key.try_emplace(keys[geometry_id], geometry_id);

    const auto live_count = static_cast<GeometryID>(keys.size());
    std::vector<bool> existed(live_count, false);
    // previous instance index -> the instance index its Retired geometries share
    std::vector<uint32_t> retired_instances(prev_geometry.instance_transforms.size(), ~0u);
    prev_geometry_remap.reserve(prev_keys.size());
    for (GeometryID prev_id = 0; prev_id < prev_keys.size(); prev_id++) {
        GeometryID geometry_id = prev_id;
        if (!same_key(prev_id)) {
            const auto it = geometry_of_key.find(prev_keys[prev_id]);
            geometry_id = it != geometry_of_key.end() ? it->second : live_count;
        }
        if (geometry_id < live_count && continues_at(geometry_id, prev_id)) {
            prev_geometry_remap.push_back({geometry_id, geometry.instance_indices[geometry_id]});
            existed[geometry_id] = true;
            continue;
        }

        // last frame's geometry under last frame's transforms
        const uint32_t prev_instance = prev_geometry.instance_indices[prev_id];
        uint32_t& instance = retired_instances[prev_instance];
        if (instance == ~0u) {
            instance = static_cast<uint32_t>(geometry.instance_transforms.size());
            const float4x4& transform = prev_geometry.instance_transforms[prev_instance];
            const float4x4& inverse_transposed =
                prev_geometry.inverse_transposed_instance_transforms[prev_instance];
            geometry.instance_transforms.push_back(transform);
            geometry.inverse_transposed_instance_transforms.push_back(inverse_transposed);
            geometry.prev_instance_transforms.push_back(transform);
            geometry.prev_inverse_transposed_instance_transforms.push_back(inverse_transposed);
        }
        GeometryData retired_geometry = prev_geometry.geometries[prev_id];
        retired_geometry.prev_vertices = retired_geometry.vertices;
        retired_geometry.flags = GeometryDataFlags(
            (retired_geometry.flags & ~GeometryDataFlags::New) | GeometryDataFlags::Retired);
        prev_geometry_remap.push_back(
            {static_cast<GeometryID>(geometry.geometries.size()), instance});
        geometry.geometries.push_back(retired_geometry);
        geometry.instance_indices.push_back(instance);
    }

    for (GeometryID geometry_id = 0; geometry_id < live_count; geometry_id++) {
        if (existed[geometry_id])
            continue;
        // the previous vertices of its mesh belong to another identity
        GeometryData& new_geometry = geometry.geometries[geometry_id];
        new_geometry.prev_vertices = new_geometry.vertices;
        new_geometry.flags = GeometryDataFlags(new_geometry.flags | GeometryDataFlags::New);
    }
}

void Scene::upload_prev_geometry_remap(const CommandBufferHandle& cmd) {
    if (prev_geometry_remap.empty())
        return;
    const vk::DeviceSize remap_size = prev_geometry_remap.size() * sizeof(GeometryRemap);
    if (!prev_geometry_remap_buffer || prev_geometry_remap_buffer->get_size() < remap_size) {
        retire(std::move(prev_geometry_remap_buffer));
        prev_geometry_remap_buffer = allocator->create_buffer(
            remap_size * 3 / 2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::NONE, "Scene::prev_geometry_remap");
        shader_object->get_cursor()["prev_geometry_remap"] = prev_geometry_remap_buffer;
    }
    allocator->get_staging()->cmd_to_device(cmd, prev_geometry_remap_buffer,
                                            prev_geometry_remap.data(), 0, remap_size);
}

namespace {

constexpr vk::BufferUsageFlags SHARED_VB_USAGE =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
    vk::BufferUsageFlagBits::eShaderDeviceAddress;

constexpr vk::BufferUsageFlags SHARED_IB_USAGE =
    SHARED_VB_USAGE | vk::BufferUsageFlagBits::eIndexBuffer;

constexpr vk::DeviceSize MIN_BUFFER_CAPACITY = vk::DeviceSize(32) * 1024 * 1024;
constexpr vk::DeviceSize RESIZE_QUANTUM = vk::DeviceSize(32) * 1024 * 1024;
constexpr vk::DeviceSize MIN_FREE_HEADROOM = vk::DeviceSize(16) * 1024 * 1024;

constexpr vk::DeviceSize size_with_headroom(const vk::DeviceSize required) {
    const vk::DeviceSize free_space = std::max(MIN_FREE_HEADROOM, required / 4);
    const vk::DeviceSize target = std::max(MIN_BUFFER_CAPACITY, required + free_space);
    return ((target + RESIZE_QUANTUM - 1) / RESIZE_QUANTUM) * RESIZE_QUANTUM;
}

} // namespace

void Scene::reallocate_shared_buffer(
    const CommandBufferHandle& cmd,
    VMAMemorySubAllocatorHandle& slot,
    vk::DeviceSize& capacity_slot,
    const std::initializer_list<MeshBufferRegion MeshInfo::*> region_fields,
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
    regions.reserve(mesh_ids.count() * region_fields.size());

    for (const MeshID id : mesh_ids) {
        for (MeshBufferRegion MeshInfo::* const region_field : region_fields) {
            MeshBufferRegion& region = mesh_infos[id].*region_field;
            if (!region)
                continue;

            const vk::DeviceSize old_offset = region.get_offset();
            const auto va = new_alloc->allocate({region.size, region.alignment, ~0u});
            regions.push_back({old_offset, va.offset, region.size});
            retire(std::move(region.suballoc));
            region.suballoc = std::make_shared<VMAMemorySubAllocation>(
                context, new_alloc, va.allocation, va.offset, region.size);
        }
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
    vertex_jobs.reserve(mesh_ids.count());
    max_vertex_count_in_jobs = 0;

    const auto drop_prev_vertices = [&](MeshInfo& info) {
        retire(std::move(info.prev_vertex_buffer.suballoc));
        info.prev_vertex_buffer = MeshBufferRegion{};
    };

    vk::DeviceSize total_vb = 0;
    vk::DeviceSize total_ib = 0;
    // allocated by this update, while the regions it retires still occupy the buffers
    vk::DeviceSize allocated_vb = 0;
    vk::DeviceSize allocated_ib = 0;
    {
        MERIAN_PROFILE_SCOPE("size_prepass");
        for (const MeshID id : mesh_ids) {
            MeshInfo& info = mesh_infos[id];
            if (info.instances.empty()) {
                drop_prev_vertices(info);
                continue;
            }
            const Mesh& mesh = *info.mesh;
            const vk::DeviceSize vb_size = mesh.get_vertex_count() * sizeof(PackedVertexData);
            const vk::DeviceSize ib_size =
                mesh.get_primitive_count() * std::size_t(3) * size_for_index_type(mesh.index_type);
            // last frame's vertex and index regions stay allocated next to the new ones
            const bool two_vertex_regions =
                info.prev_vertex_buffer || mesh.vertices_dirty || info.is_dynamic();
            total_vb += two_vertex_regions ? 2 * vb_size : vb_size;
            total_ib += mesh.indices_dirty ? 2 * ib_size : ib_size;
            if ((mesh.vertices_dirty || info.is_dynamic()) &&
                info.prev_vertex_buffer.size != vb_size)
                allocated_vb += vb_size;
            if (mesh.indices_dirty)
                allocated_ib += ib_size;
        }
    }

    const auto ensure_capacity =
        [&](VMAMemorySubAllocatorHandle& slot, vk::DeviceSize& capacity,
            const std::initializer_list<MeshBufferRegion MeshInfo::*> region_fields,
            const vk::DeviceSize total, const vk::DeviceSize allocated,
            const vk::BufferUsageFlags usage, const std::string& debug_name) {
            const vk::DeviceSize used = slot ? capacity - slot->get_free_size() : 0;
            if (total <= capacity && used + allocated <= capacity)
                return;
            MERIAN_PROFILE_SCOPE_GPU(cmd, "reallocate_shared_buffer");
            reallocate_shared_buffer(cmd, slot, capacity, region_fields,
                                     size_with_headroom(std::max(total, used + allocated)), usage,
                                     debug_name);
        };
    ensure_capacity(shared_vb_suballoc, shared_vb_capacity,
                    {&MeshInfo::vertex_buffer, &MeshInfo::prev_vertex_buffer}, total_vb,
                    allocated_vb, SHARED_VB_USAGE, "Scene::shared_vb");
    ensure_capacity(shared_ib_suballoc, shared_ib_capacity, {&MeshInfo::index_buffer}, total_ib,
                    allocated_ib, SHARED_IB_USAGE, "Scene::shared_ib");

    // No-op if region already matches; reused regions stay where they are.
    const auto ensure_region = [&](MeshBufferRegion& region,
                                   const VMAMemorySubAllocatorHandle& slot,
                                   const vk::DeviceSize size, const vk::DeviceSize alignment) {
        if (region && region.size == size && region.alignment == alignment)
            return;
        retire(std::move(region.suballoc));
        const auto va = slot->allocate({size, alignment, ~0u});
        region.suballoc =
            std::make_shared<VMAMemorySubAllocation>(context, slot, va.allocation, va.offset, size);
        region.size = size;
        region.alignment = alignment;
    };

    static_assert(alignof(PackedVertexData) <= 16,
                  "StagingMemoryManager aligns every upload to 16");
    struct StagedRange {
        vk::DeviceAddress addr;
        BufferHandle buffer;
        vk::DeviceSize buffer_offset;
    };
    const auto stage = [&](const vk::DeviceSize size, const auto& fill) -> StagedRange {
        BufferHandle buffer;
        vk::DeviceSize buffer_offset = 0;
        const MemoryAllocationHandle memory =
            allocator->get_staging()->get_upload_staging_space(size, buffer, buffer_offset);
        cmd->keep_until_pool_reset(buffer);
        fill(memory->map());
        memory->unmap();
        return {buffer->get_device_address() + buffer_offset, buffer, buffer_offset};
    };

    // Batch one cmd->copy per distinct source buffer.
    std::unordered_map<Buffer*, std::pair<BufferHandle, std::vector<vk::BufferCopy>>> index_copies;

    {
        MERIAN_PROFILE_SCOPE("fill_staging");
        for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
            MeshGroup& group = mesh_groups[group_id];
            const bool pretransform_group =
                group.is_pretransformed(mesh_infos, pretransform_animated);
            // Pretransformed animated meshes must be re-pretransformed when the node moved.
            const bool check_node_transform = pretransform_group && group.has_animated_node;

            for (const MeshID mesh_id : group.meshes) {
                MeshInfo& info = mesh_infos[mesh_id];
                Mesh& mesh = *info.mesh;

                if (mesh.indices_dirty || (mesh.vertices_dirty && !mesh.is_morphed()))
                    info.uv_version++;

                if (check_node_transform) {
                    mesh.vertices_dirty |= scene_graph[*info.instances.begin()]->transform_dirty;
                }
                // vertices that are not uploaded are also last frame's
                if (!mesh.vertices_dirty)
                    drop_prev_vertices(info);
                if (!mesh.is_dirty())
                    continue;

                const uint32_t vertex_count = mesh.get_vertex_count();
                const uint32_t primitive_count = mesh.get_primitive_count();
                const vk::DeviceSize vb_size = vertex_count * sizeof(PackedVertexData);
                const vk::DeviceSize ib_size =
                    primitive_count * std::size_t(3) * size_for_index_type(mesh.index_type);

                // a topology change is a new identity
                const bool topology_changed = mesh.indices_dirty ||
                                              vb_size != info.vertex_buffer.size ||
                                              primitive_count != info.primitive_count;
                if (topology_changed)
                    info.serial = next_serial++;
                info.primitive_count = primitive_count;

                group.blas_dirty = true;
                if (topology_changed)
                    group.cached_blas_size_info.reset();

                frame_stats.vertices_uploaded += vertex_count;
                frame_stats.indices_uploaded += primitive_count;
                frame_stats.upload_bytes += vb_size + ib_size;

                if (mesh.vertices_dirty) {
                    // last frame's vertices become the previous ones, the region before is reused
                    std::swap(info.vertex_buffer, info.prev_vertex_buffer);
                    ensure_region(info.vertex_buffer, shared_vb_suballoc, vb_size,
                                  alignof(PackedVertexData));
                    if (topology_changed || starting_run)
                        drop_prev_vertices(info);

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
                                vertex_src_addr = stage(vb_size, [&](void* dst) {
                                                      std::memcpy(dst, src.data, vb_size);
                                                  }).addr;
                            } else {
                                static_assert(std::is_same_v<T, Mesh::HostVertices>);
                                frame_stats.meshes_uploaded_host_unpacked++;
                                vertex_src_addr =
                                    stage(vb_size, [&](void* dst) {
                                        src->write(static_cast<PackedVertexData*>(dst));
                                    }).addr;
                            }
                        },
                        mesh.get_vertices());

                    const NodeID node_id = *info.instances.begin();
                    const float4x4& M = get_global_transform(node_id);
                    const bool pretransform_mesh = pretransform_group && M != identity();
                    vertex_jobs.push_back(TransformVertexJob{
                        .src = vertex_src_addr,
                        .dst = info.vertex_buffer.get_device_address(),
                        .transform = pretransform_mesh ? M : float4x4(identity()),
                        .inverse_transposed = pretransform_mesh
                                                  ? get_global_inverse_transposed_transform(node_id)
                                                  : float4x4(identity()),
                        .vertex_count = vertex_count,
                        ._pad = {},
                    });
                    max_vertex_count_in_jobs = std::max(max_vertex_count_in_jobs, vertex_count);
                    frame_stats.gpu_vertex_transforms++;
                    frame_stats.gpu_vertex_transform_vertices += vertex_count;

                    mesh.vertices_dirty = false;
                }

                if (mesh.indices_dirty) {
                    // never overwritten in place: retired geometry may read the old indices
                    retire(std::move(info.index_buffer.suballoc));
                    info.index_buffer = MeshBufferRegion{};
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
                                const auto staged = stage(ib_size, [&](void* dst) {
                                    std::memcpy(dst, src.data, ib_size);
                                });
                                auto& entry = index_copies[staged.buffer.get()];
                                entry.first = staged.buffer;
                                entry.second.push_back({staged.buffer_offset, dst_offset, ib_size});
                            } else {
                                static_assert(std::is_same_v<T, Mesh::HostIndices>);
                                const auto staged =
                                    stage(ib_size, [&](void* dst) { src->write(dst); });
                                auto& entry = index_copies[staged.buffer.get()];
                                entry.first = staged.buffer;
                                entry.second.push_back({staged.buffer_offset, dst_offset, ib_size});
                            }
                        },
                        mesh.get_indices());
                }
                mesh.indices_dirty = false;
            }
        }
    }

    if (!index_copies.empty()) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "index_copy");
        // Source buffers may have been written earlier in the same cmd; sync transfer write→read.
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
}

void Scene::ensure_micromaps(Submission& submission,
                             const ShaderObjectAllocatorHandle& obj_allocator,
                             const bool complete) {
    if (!micromaps) {
        return;
    }

    const bool opacity = opacity_micromaps_enabled && Micromaps::is_opacity_supported(context);
    const TextureManager& textures = *material_system->get_texture_manager();
    const auto is_loaded = [&](const TextureID texture_id) {
        return texture_id < textures.get_capacity() && textures.get_texture(texture_id);
    };
    std::vector<Micromaps::MeshGeometry> baked;
    for (MeshID mesh_id = 0; mesh_id < mesh_infos.size(); mesh_id++) {
        const MeshInfo& info = mesh_infos[mesh_id];
        if (!info.mesh || !info.vertex_buffer) {
            continue;
        }
        const Mesh& mesh = *info.mesh;
        if (mesh.has_variable_topology() || mesh.get_primitive_count() == 0) {
            continue;
        }
        const bool alpha_tested =
            !mesh.is_opaque() && is_loaded(material_system->get_alpha_texture_id(mesh.material_id));
        const bool emissive = emission_micromaps_enabled && !(mesh.flags & MeshFlags::UseEnvMap) &&
                              material_system->is_emissive(mesh.material_id) &&
                              is_loaded(material_system->get_emission_texture_id(mesh.material_id));
        if (!(opacity && alpha_tested) && !emissive) {
            continue;
        }

        Micromaps::MeshGeometry& geometry = baked.emplace_back();
        geometry.mesh_id = mesh_id;
        geometry.mesh_serial = info.serial;
        geometry.alpha_tested = alpha_tested;
        geometry.opacity = opacity && alpha_tested;
        geometry.emission = emissive;
        geometry.geometry.material_id = mesh.material_id;
        geometry.primitive_count = mesh.get_primitive_count();
        geometry.geometry.vertices = info.vertex_buffer.get_device_address();
        geometry.geometry.indices =
            info.index_buffer ? info.index_buffer.get_device_address() : vk::DeviceAddress{0};
        geometry.geometry.flags = index_type_flag(mesh.index_type);
        geometry.uv_version = info.uv_version;

        const Mesh::MeshVertexData vertices = mesh.get_vertices();
        const Mesh::MeshIndexData indices = mesh.get_indices();
        const auto* host_vertices = std::get_if<Mesh::HostPacked<PackedVertexData>>(&vertices);
        const auto* host_indices = std::get_if<Mesh::HostPacked<void>>(&indices);
        if (host_vertices != nullptr && (host_indices != nullptr || !mesh.has_indices())) {
            geometry.host_vertices = host_vertices->data;
            geometry.host_indices = host_indices != nullptr ? host_indices->data : nullptr;
            geometry.host_index_type = mesh.index_type;
        }
    }

    std::vector<MeshID> changed;
    micromaps->update(submission, baked, material_system, changed, obj_allocator, complete);

    for (const MeshID mesh_id : changed) {
        if (mesh_id >= mesh_to_group.size() || mesh_to_group[mesh_id] == MESH_GROUP_ID_INVALID) {
            continue;
        }
        MeshGroup& group = mesh_groups[mesh_to_group[mesh_id]];
        retire(std::move(group.blas));
        retire(std::move(group.spare_blas));
        group.cached_blas_size_info.reset();
        group.blas_dirty = true;
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
            // the previous frame's BLAS is the current one again once a frame passed unchanged
            retire(std::move(group.spare_blas));
            continue;
        }
        tlas_dirty = true;
        std::swap(group.blas, group.spare_blas);

        auto& blas_geometry = blas_geometries[group_id];
        blas_geometry.micromaps.reserve(group.meshes.size());

        for (MeshID mesh_id : group.meshes) {
            const MeshInfo& info = mesh_infos[mesh_id];
            const Mesh& mesh = *info.mesh;

            vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
            triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
            triangles.vertexData = info.vertex_buffer.get_device_address();
            triangles.vertexStride = sizeof(PackedVertexData);
            triangles.maxVertex = mesh.get_vertex_count() - 1;
            triangles.indexType = mesh.index_type;
            triangles.indexData =
                info.index_buffer ? info.index_buffer.get_device_address() : vk::DeviceAddress{0};

            vk::AccelerationStructureGeometryKHR geom;
            geom.geometryType = vk::GeometryTypeKHR::eTriangles;
            geom.geometry.triangles = triangles;

            if (mesh.flags & MeshFlags::IsOpaque) {
                geom.flags = vk::GeometryFlagBitsKHR::eOpaque;
            } else if (micromaps) {
                vk::AccelerationStructureTrianglesOpacityMicromapEXT omm;
                if (micromaps->get_opacity(mesh_id, omm)) {
                    geom.geometry.triangles.pNext = &blas_geometry.micromaps.emplace_back(omm);
                }
            }

            vk::AccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = mesh.get_primitive_count();

            blas_geometry.geometries.push_back(geom);
            blas_geometry.ranges.push_back(range);
        }

        const bool can_update =
            group.spare_blas &&
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
            retire(std::move(group.blas));
            group.blas = allocator->create_acceleration_structure(
                vk::AccelerationStructureTypeKHR::eBottomLevel, size_info,
                fmt::format("Scene::blas[{}]", group_id));
        }

        as_builder.queue_build(blas_geometry.geometries, blas_geometry.ranges, group.blas,
                               size_info, group.blas_build_flags);
        group.blas_dirty = false;
        group.blas_last_built_update = update_count;
        group.blas_last_updated_update = update_count;
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
                          return mesh_groups[a].blas_last_built_update <
                                 mesh_groups[b].blas_last_built_update;
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

            const bool outgrown =
                !group.blas || group.blas->get_size() < size_info.accelerationStructureSize;
            if (outgrown) {
                retire(std::move(group.blas));
                group.blas = allocator->create_acceleration_structure(
                    vk::AccelerationStructureTypeKHR::eBottomLevel, size_info,
                    fmt::format("Scene::blas[{}]", group_id));
            }
            if (i < rebuild_count ||
                group.spare_blas->get_size() < size_info.accelerationStructureSize) {
                as_builder.queue_build(blas_geometry.geometries, blas_geometry.ranges, group.blas,
                                       size_info, group.blas_build_flags);
                group.blas_last_built_update = update_count;
                frame_stats.blas_builds++;
                frame_stats.blas_builds_dynamic++;
            } else {
                as_builder.queue_update(blas_geometry.geometries, blas_geometry.ranges,
                                        group.spare_blas, group.blas, size_info,
                                        group.blas_build_flags);
                frame_stats.blas_updates++;
            }
            group.blas_dirty = false;
            group.blas_last_updated_update = update_count;
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
            tlas_instance.mask = group.instance_mask;
            tlas_instance.accelerationStructureReference =
                group.blas->get_acceleration_structure_device_address();

            if (group.is_pretransformed(mesh_infos, pretransform_animated)) {
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
            if (group.flags & MeshFlags::FlipFacing) {
                geometry_instance_flags |= vk::GeometryInstanceFlagBitsKHR::eTriangleFlipFacing;
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
        retire(std::move(tlas_instances_buffer));
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
    AccelerationStructureHandle& tlas = geometry.tlas;
    if (!tlas || tlas->get_size() < size_info.accelerationStructureSize) {
        retire(std::move(tlas));
        tlas = allocator->create_acceleration_structure(vk::AccelerationStructureTypeKHR::eTopLevel,
                                                        size_info);
    }

    as_builder.queue_build(tlas_instances.size(), tlas_instances_buffer, tlas, size_info);
    as_builder.get_cmds_tlas(cmd, as_scratch_buffer);
}

void Scene::update(Submission& submission,
                   const float time,
                   const float time_diff,
                   const uint32_t frame,
                   const uint32_t seed,
                   const ShaderObjectAllocatorHandle& obj_allocator_in) {
    // follows the submissions the micromap bake splits off
    const CommandBufferHandle& cmd = submission.get_cmd();
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::update");
    update_count++;
    starting_run |= frame == 0;
    frame_stats = {};
    last_updates = 0;

    // the previous frame may still read what this update overwrites
    cmd->barrier(vk::MemoryBarrier2{vk::PipelineStageFlagBits2::eAllCommands,
                                    {},
                                    vk::PipelineStageFlagBits2::eAllCommands,
                                    {}});

    if (!obj_allocator_in && !fallback_obj_allocator) {
        fallback_obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);
    }
    const ShaderObjectAllocatorHandle& obj_allocator =
        obj_allocator_in ? obj_allocator_in : fallback_obj_allocator;

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "on_update");
        on_update(cmd, time, time_diff, frame);
    }

    process_pending_env_load(cmd);

    if (!is_ready()) {
        // retired objects stay alive for the next ready update, which reads them as previous
        get_texture_manager()->upload(cmd);
        return;
    }

    if (starting_run) {
        prev_env_map = env_map->clone();
    }
    compose_prev_env();

    // rebuild once here so the per-frame uploads below write into the current object
    shader_object.get();

    assert(!cameras.empty() &&
           "the scene implementation must ensure that there is at least one camera");

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "material_system::update");
        material_system->update(cmd);
    }

    std::swap(geometry, prev_geometry);

    // Must run before upload_meshes, which clears the dirty flags.
    if (needs_regroup) {
        compute_mesh_groups();
    }

    // Must precede upload_geometry_data so device addresses are available.
    upload_meshes(cmd);

    if (!vertex_jobs.empty()) {
        stage_transform_jobs(cmd);

        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        });

        dispatch_batched_transforms(cmd);
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    });

    if (as_supported) {
        // binds the material system: after its writes, before the geometry data's micromaps
        ensure_micromaps(submission, obj_allocator, starting_run);
    }

    upload_geometry_data(cmd);

    frame_stats.geometry_data_bytes = geometry.geometries.size() * sizeof(GeometryData);
    frame_stats.transform_data_bytes = geometry.instance_transforms.size() * sizeof(float4x4) * 4;

    if (as_supported) {
        build_blas(cmd);

        tlas_dirty |= needs_regroup || transforms_changed;
        if (!tlas_dirty && prev_geometry.tlas) {
            retire(std::move(geometry.tlas));
            geometry.tlas = prev_geometry.tlas;
        } else {
            // the previous frame reads its TLAS, build into the one from two frames ago
            if (geometry.tlas == prev_geometry.tlas) {
                geometry.tlas.reset();
            }
            build_tlas(cmd);
            tlas_dirty = false;
            frame_stats.tlas_rebuilt = true;
            frame_stats.tlas_instance_count = static_cast<uint32_t>(tlas_instances.size());
            frame_stats.tlas_instance_data_bytes =
                tlas_instances.size() * sizeof(vk::AccelerationStructureInstanceKHR);
        }
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer |
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    });

    for (const NodeID& nid : node_ids) {
        Node& node = *scene_graph[nid];
        node.transform_dirty = false;
    }
    if (needs_regroup)
        last_updates |= SceneUpdateGeometry;
    if (transforms_changed)
        last_updates |= SceneUpdateTransform;
    needs_regroup = false;
    transforms_changed = false;

    const auto cam = get_active_camera();
    assert(cam);

    const bool run_start = starting_run;
    if (starting_run) {
        prev_active_camera = *cam;
        prev_time.reset();
        lights.start_run();
        starting_run = false;
    }

    // the thin lens hint must be settled before the shader object is fetched below
    if (const bool thin_lens = cam->get_aperture_radius() > 0.f; thin_lens != thin_lens_enabled) {
        set_enable_thin_lens(thin_lens);
    }
    lights.update_constants(composition);

    auto c = shader_object->get_cursor();

    if (as_supported) {
        c["as"]["as"] = geometry.tlas;
        c["prev_as"]["as"] = prev_geometry.tlas && !run_start ? prev_geometry.tlas : geometry.tlas;
    }
    c["frame"] = frame;
    c["seed"] = seed;
    c["prev_geometry_remap_count"] = static_cast<uint32_t>(prev_geometry_remap.size());
    const float scene_time = get_time(time);
    c["time"] = scene_time;
    c["time_diff"] = time_diff;
    c["prev_time"] = prev_time.value_or(scene_time);
    prev_time = scene_time;
    exterior_volume->write_to(c["exterior_volume"]);

    // prev_active_camera still holds last frame's pose here (updated below).
    if (!(*cam == prev_active_camera))
        last_updates |= SceneUpdateCamera;

    cam->advance_jitter(frame, seed);

    prev_active_camera.write_to(c["prev_camera"]);
    cam->write_to(c["camera"]);
    prev_active_camera = *cam;

    env_map->write_to(c["env_map"]);
    prev_env_map->write_to(c["prev_env_map"]);
    prev_env_map = env_map->clone();

    // binds the scene object: every write to it must precede this
    lights.set_env_emissive(env_map->is_emissive());
    lights.set_env_state(env_map->get_version(), env_map->is_static());
    lights.set_camera(cam->get_position());
    lights.prepare(cmd);
    lights.write_to(c["nee"]);
    lights.write_prev_to(c["prev_nee"]);
    lights.update(cmd, composition, shader_object.get(), obj_allocator);

    drain_retired(cmd);
}

void Scene::drain_retired(const CommandBufferHandle& cmd) {
    frame_stats.objects_retired = static_cast<uint32_t>(retired.size());
    for (ConstObjectHandle& object : retired) {
        cmd->keep_until_pool_reset(std::move(object));
    }
    retired.clear();
}

std::vector<float3> compute_vertex_normals(const std::vector<float3>& positions,
                                           const std::vector<uint3>& triangles) {
    std::vector<float3> normals(positions.size(), float3(0));
    for (const uint3& tri : triangles) {
        const float3 face =
            cross(positions[tri.y] - positions[tri.x], positions[tri.z] - positions[tri.x]);
        normals[tri.x] += face;
        normals[tri.y] += face;
        normals[tri.z] += face;
    }
    for (float3& n : normals) {
        n = length(n) > 1e-20f ? normalize(n) : float3(0, 1, 0);
    }
    return normals;
}

} // namespace merian
