#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/math.hpp"

#include "pretransform_prev_vertex.slang.spv.h"
#include "pretransform_vertex.slang.spv.h"

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
      material_system(material_system), thread_pool(std::make_shared<ThreadPool>()) {

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

    Mesh& mesh = *meshes[mesh_id];

    if (mesh.instances.size() == 1 && (!mesh.is_dynamic() || pretransform_animated)) {
        mesh.vertices_dirty = true;
    }
    mesh.instances.insert(node_id);
    if (scene_graph[node_id]->is_animated)
        mesh.animated_instance_count++;
    needs_regroup = true;
}

void Scene::remove_mesh_instance(const MeshID mesh_id, const NodeID node_id) {
    assert(mesh_ids.is_used(mesh_id));
    assert(node_ids.is_used(node_id));

    Mesh& mesh = *meshes[mesh_id];

    if (mesh.instances.erase(node_id) > 0) {
        if (scene_graph[node_id]->is_animated) {
            assert(mesh.animated_instance_count > 0);
            mesh.animated_instance_count--;
        }
        if (mesh.instances.size() == 1 && (!mesh.is_dynamic() || pretransform_animated)) {
            mesh.vertices_dirty = true;
        }
        needs_regroup = true;
    }
}

void Scene::remove_mesh(const MeshID mesh_id) {
    assert(mesh_ids.is_used(mesh_id));

    if (mesh_id < mesh_to_group.size() && mesh_to_group[mesh_id] != MESH_GROUP_ID_INVALID) {
        MeshGroup& group = mesh_groups[mesh_to_group[mesh_id]];
        group.cached_blas_size_info.reset();
        group.blas.reset();
    }

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
        if (meshes[mesh_id]->instances.erase(node_id) > 0 && node.is_animated) {
            assert(meshes[mesh_id]->animated_instance_count > 0);
            meshes[mesh_id]->animated_instance_count--;
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
// GPU pretransform
// ---------------------------------------------------------------------------

namespace {

struct PretransformVertexPushConstant {
    float4x4 transform;
    float4x4 inverse_transposed;
    uint32_t vertex_count;
};

struct PretransformPrevVertexPushConstant {
    float4x4 transform;
    uint32_t vertex_count;
};

constexpr uint32_t PRETRANSFORM_LOCAL_SIZE_X = 64;

} // namespace

void Scene::ensure_pretransform_pipelines() {
    if (pretransform_vertex_pipeline && pretransform_prev_vertex_pipeline) {
        return;
    }

    pretransform_descriptor_layout =
        DescriptorSetLayoutBuilder()
            .add_binding_storage_buffer(1, vk::ShaderStageFlagBits::eCompute)
            .add_binding_storage_buffer(1, vk::ShaderStageFlagBits::eCompute)
            .build_push_descriptor_layout(context);

    auto vertex_module = EntryPoint::create(context, merian_pretransform_vertex_slang_spv(),
                                            merian_pretransform_vertex_slang_spv_size(), "main",
                                            vk::ShaderStageFlagBits::eCompute);
    auto prev_module = EntryPoint::create(context, merian_pretransform_prev_vertex_slang_spv(),
                                          merian_pretransform_prev_vertex_slang_spv_size(), "main",
                                          vk::ShaderStageFlagBits::eCompute);

    auto vertex_layout =
        PipelineLayoutBuilder(context)
            .add_descriptor_set_layout(pretransform_descriptor_layout)
            .add_push_constant<PretransformVertexPushConstant>(vk::ShaderStageFlagBits::eCompute)
            .build_pipeline_layout();
    auto prev_layout = PipelineLayoutBuilder(context)
                           .add_descriptor_set_layout(pretransform_descriptor_layout)
                           .add_push_constant<PretransformPrevVertexPushConstant>(
                               vk::ShaderStageFlagBits::eCompute)
                           .build_pipeline_layout();

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(PRETRANSFORM_LOCAL_SIZE_X);
    SpecializationInfoHandle spec = spec_builder.build();

    pretransform_vertex_pipeline = ComputePipeline::create(vertex_layout, vertex_module, spec);
    pretransform_prev_vertex_pipeline = ComputePipeline::create(prev_layout, prev_module, spec);
}

void Scene::pretransform_vertices_gpu(const CommandBufferHandle& cmd,
                                      const BufferHandle& src,
                                      const BufferHandle& dst,
                                      const float4x4& transform,
                                      const float4x4& inverse_transposed,
                                      const uint32_t vertex_count,
                                      const vk::DeviceSize src_offset) {
    ensure_pretransform_pipelines();

    PretransformVertexPushConstant pc{};
    pc.transform = transform;
    pc.inverse_transposed = inverse_transposed;
    pc.vertex_count = vertex_count;

    cmd->bind(pretransform_vertex_pipeline);
    cmd->push_descriptor_set(pretransform_vertex_pipeline, src->get_descriptor_info(src_offset),
                             dst->get_descriptor_info());
    cmd->push_constant(pretransform_vertex_pipeline, pc);
    cmd->dispatch((vertex_count + PRETRANSFORM_LOCAL_SIZE_X - 1) / PRETRANSFORM_LOCAL_SIZE_X, 1, 1);
}

void Scene::pretransform_prev_vertices_gpu(const CommandBufferHandle& cmd,
                                           const BufferHandle& src,
                                           const BufferHandle& dst,
                                           const float4x4& transform,
                                           const uint32_t vertex_count,
                                           const vk::DeviceSize src_offset) {
    ensure_pretransform_pipelines();

    PretransformPrevVertexPushConstant pc{};
    pc.transform = transform;
    pc.vertex_count = vertex_count;

    cmd->bind(pretransform_prev_vertex_pipeline);
    cmd->push_descriptor_set(pretransform_prev_vertex_pipeline,
                             src->get_descriptor_info(src_offset), dst->get_descriptor_info());
    cmd->push_constant(pretransform_prev_vertex_pipeline, pc);
    cmd->dispatch((vertex_count + PRETRANSFORM_LOCAL_SIZE_X - 1) / PRETRANSFORM_LOCAL_SIZE_X, 1, 1);
}

void Scene::set_pretransform_animated(bool value) {
    if (pretransform_animated == value)
        return;

    pretransform_animated = value;

    needs_regroup = true;
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
                const Mesh& mesh = *meshes[id];
                const MeshGroupID group_id =
                    id < mesh_to_group.size() ? mesh_to_group[id] : MESH_GROUP_ID_INVALID;
                const std::string group_label = group_id == MESH_GROUP_ID_INVALID
                                                    ? "no group"
                                                    : fmt::format("group {}", group_id);
                if (props.st_begin_child(
                        fmt::format("mesh_{:04} ", id),
                        fmt::format("{:04}: {} [{}]", id, mesh.name, group_label))) {
                    props.output_text("{}", mesh);
                    props.st_end_child();
                }
            }
            props.st_end_child();
        }

        if (props.st_begin_child("mesh_groups", "Mesh Groups")) {
            for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
                const MeshGroup& group = mesh_groups[group_id];
                if (props.st_begin_child(fmt::format("group_{:04}", group_id),
                                         fmt::format("{:04}: {} ({} mesh{})", group_id, group.flags,
                                                     group.meshes.size(),
                                                     group.meshes.size() == 1 ? "" : "es"))) {
                    const std::size_t instances =
                        group.meshes.empty() ? 0 : group.get_instances(meshes).size();
                    const bool pretransformed =
                        !group.meshes.empty() &&
                        group.is_pretranformed(meshes, pretransform_animated);
                    std::size_t group_vertices = 0;
                    std::size_t group_triangles = 0;
                    for (const MeshID mesh_id : group.meshes) {
                        group_vertices += meshes[mesh_id]->get_vertex_count();
                        group_triangles += meshes[mesh_id]->get_primitive_count();
                    }
                    props.output_text(
                        "split flags: {}\nhas animated node: {}\nhas morphed mesh: "
                        "{}\nhas variable topology: {}\nall opaque: {}\npretransformed: "
                        "{}\ninstances: "
                        "{}\nvertices: {}\ntriangles: {}\nblas: {}\nblas dirty: {}\nblas last "
                        "built frame: {}",
                        group.flags, group.has_animated_node, group.has_morphed_mesh,
                        group.has_variable_topology_mesh, group.all_opaque, pretransformed,
                        instances, group_vertices, group_triangles,
                        group.blas ? fmt::format("{} bytes", format_size(group.blas->get_size()))
                                   : "<none>",
                        group.blas_dirty, group.blas_last_built_frame);

                    if (props.st_begin_child("members", "Members")) {
                        for (const MeshID mesh_id : group.meshes) {
                            props.output_text("{:04}: {}", mesh_id, meshes[mesh_id]->name);
                        }
                        props.st_end_child();
                    }
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
        bool pretransform = get_pretransform_animated();
        if (props.config_bool("Pretransform Animated", pretransform)) {
            set_pretransform_animated(pretransform);
        }
        props.st_end_child();
    }

    if (props.st_begin_child("stats", "Statistics")) {
        std::size_t total_vertices = 0;
        std::size_t total_triangles = 0;
        std::size_t total_instances = 0;
        std::size_t morphed_meshes = 0;
        std::size_t dynamic_meshes = 0;
        for (const MeshID id : mesh_ids) {
            const Mesh& mesh = *meshes[id];
            total_vertices += mesh.get_vertex_count();
            total_triangles += mesh.get_primitive_count();
            total_instances += mesh.instances.size();
            if (mesh.is_morphed()) {
                morphed_meshes++;
            }
            if (mesh.is_dynamic()) {
                dynamic_meshes++;
            }
        }

        props.output_text(
            "nodes: {}\nmeshes: {} ({} dynamic, {} morphed)\ninstances: {}\nmesh groups: "
            "{}\nvertices: {}\ntriangles: {}\nmaterials: {}\ntextures: {}\ntlas instances: "
            "{}\npending buffer releases: {}\nneeds regroup: {}\ntransforms changed: "
            "{}\npretransform dynamic: {}",
            node_ids.count(), mesh_ids.count(), dynamic_meshes, morphed_meshes, total_instances,
            mesh_groups.size(), total_vertices, total_triangles,
            material_system->get_material_count(), get_texture_manager()->get_texture_count(),
            tlas_instances.size(), pending_buffer_releases.size(), needs_regroup,
            transforms_changed, pretransform_animated);

        if (aabb.is_valid()) {
            props.output_text("aabb: min={}, max={}, size={}", aabb.get_min(), aabb.get_max(),
                              aabb.get_max() - aabb.get_min());
        } else {
            props.output_text("aabb: <not available>");
        }

        props.st_separate("Per Frame");
        props.output_text(
            "meshes uploaded: {} (device_local: {}, staged: {}, host_packed: {}, "
            "host_unpacked: {})\n"
            "vertices uploaded: {}, primitives uploaded: {}\n"
            "upload size: {}\n"
            "pretransforms cpu: {} ({} verts), gpu: {} ({} verts)\n"
            "blas builds: {} (static: {}, dynamic: {})\n"
            "tlas rebuilt: {} ({} instances)\n"
            "buffers allocated: {}, released: {}\n"
            "geometry data: {}, transforms: {}, tlas instances: {}",
            frame_stats.meshes_uploaded(), frame_stats.meshes_uploaded_device_local,
            frame_stats.meshes_uploaded_device_staged, frame_stats.meshes_uploaded_host_packed,
            frame_stats.meshes_uploaded_host_unpacked, frame_stats.vertices_uploaded,
            frame_stats.indices_uploaded, format_size(frame_stats.upload_bytes),
            frame_stats.cpu_pretransforms, frame_stats.cpu_pretransform_vertices,
            frame_stats.gpu_pretransforms, frame_stats.gpu_pretransform_vertices,
            frame_stats.blas_builds, frame_stats.blas_builds_static,
            frame_stats.blas_builds_dynamic, frame_stats.tlas_rebuilt,
            frame_stats.tlas_instance_count, frame_stats.buffers_allocated,
            frame_stats.buffers_released, format_size(frame_stats.geometry_data_bytes),
            format_size(frame_stats.transform_data_bytes),
            format_size(frame_stats.tlas_instance_data_bytes));

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
    std::map<std::set<NodeID>, std::unordered_map<uint32_t, MeshGroupID>> groups_instanced;

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
        const Mesh& mesh = *meshes[mesh_id];

        if (mesh.instances.empty()) {
            continue;
        }

        const uint32_t key = split_key(mesh.flags);
        MeshGroupID group_id;

        if (mesh.instances.size() > 1) {
            group_id = get_or_create(groups_instanced[mesh.instances], key);
        } else {
            assert(mesh.instances.size() == 1);
            const NodeID node_id = *mesh.instances.begin();
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
        group.has_animated_node |= mesh.animated_instance_count > 0;
        group.has_morphed_mesh |= mesh.is_morphed();
        group.has_variable_topology_mesh |= mesh.has_variable_topology();
        group.all_opaque &= mesh.is_opaque();
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
                group.cached_blas_size_info = prev_group.cached_blas_size_info;
                // blas_dirty is computed later when we upload the meshes, because this method is
                // not run every frame. We later also check if the blas can be actually reused or
                // if a new one is necessary.
            }
        }
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

        const bool pretransform = group.is_pretranformed(meshes, pretransform_animated);

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
}

void Scene::upload_geometry_data(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::upload_geometry_data");
    geometries.clear();

    const float4x4 identity_transform = identity();

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        assert(!group.meshes.empty());

        const bool pretransform = group.is_pretranformed(meshes, pretransform_animated);

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
                assert(!mesh.is_morphed() || prev_vertex_buffers[mesh_id]);
                gd.prev_vertices = prev_vertex_buffers[mesh_id]
                                       ? prev_vertex_buffers[mesh_id]->get_device_address()
                                       : vk::DeviceAddress{0};
                gd.flags = GeometryDataFlags{};
                if (pretransform || transform_is_identity) {
                    gd.flags = GeometryDataFlags(gd.flags | GeometryDataFlags::Pretransformed);
                }
                if (mesh.is_morphed()) {
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

void Scene::upload_meshes(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::upload_meshes");

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
            frame_stats.buffers_allocated++;
        }
    };

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        const bool pretransform_group = group.is_pretranformed(meshes, pretransform_animated);
        // Pretransformed animated meshes must be re-pretransformed when the node moved.
        const bool check_node_transform = pretransform_group && group.has_animated_node;

        for (const MeshID mesh_id : group.meshes) {
            Mesh& mesh = *meshes[mesh_id];

            mesh.vertices_dirty |=
                check_node_transform && scene_graph[*mesh.instances.begin()]->transform_dirty;

            if (!mesh.is_dirty())
                continue;

            MERIAN_PROFILE_SCOPE_GPU_DETAILED(
                cmd, fmt::format("Scene::upload_meshes::mesh[{}]", mesh_id));

            group.blas_dirty = true;
            if (mesh.has_variable_topology()) {
                group.cached_blas_size_info.reset();
            }

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

            frame_stats.vertices_uploaded += vertex_count;
            frame_stats.indices_uploaded += primitive_count;
            frame_stats.upload_bytes += vb_size + ib_size;
            if (mesh.is_morphed())
                frame_stats.upload_bytes += prev_vb_size;

            const NodeID node_id = *mesh.instances.begin();
            const float4x4& global_transform = get_global_transform(node_id);
            const float4x4& global_inverse_transposed_transform =
                get_global_inverse_transposed_transform(node_id);

            const bool pretransform_mesh = pretransform_group && global_transform != identity();

            std::visit(
                [&](auto&& src) {
                    using T = std::decay_t<decltype(src)>;
                    if constexpr (std::is_same_v<T, Mesh::DeviceLocal>) {
                        frame_stats.meshes_uploaded_device_local++;
                        if (pretransform_mesh) {
                            frame_stats.gpu_pretransforms++;
                            frame_stats.gpu_pretransform_vertices += vertex_count;
                        }
                    } else if constexpr (std::is_same_v<T, Mesh::DeviceStaged>) {
                        frame_stats.meshes_uploaded_device_staged++;
                        if (pretransform_mesh) {
                            frame_stats.gpu_pretransforms++;
                            frame_stats.gpu_pretransform_vertices += vertex_count;
                        }
                    } else if constexpr (std::is_same_v<T, Mesh::HostPacked<PackedVertexData>>) {
                        frame_stats.meshes_uploaded_host_packed++;
                        if (pretransform_mesh) {
                            frame_stats.cpu_pretransforms++;
                            frame_stats.cpu_pretransform_vertices += vertex_count;
                        }
                    } else {
                        static_assert(std::is_same_v<T, Mesh::HostVertices>);
                        frame_stats.meshes_uploaded_host_unpacked++;
                        if (pretransform_mesh) {
                            frame_stats.cpu_pretransforms++;
                            frame_stats.cpu_pretransform_vertices += vertex_count;
                        }
                    }
                },
                mesh.get_vertices());

            // Dispatches upload for a mesh data variant.
            // cpu_write:        (src, void*) -> writes raw host data
            // cpu_pretransform: (src, void*) -> writes pretransformed host data (nullptr to skip)
            // gpu_pretransform: (BufferHandle src, BufferHandle dst, DeviceSize src_offset) ->
            // compute pretransform (nullptr to skip)
            const auto upload = [&](auto&& variant, BufferHandle& dst, vk::DeviceSize size,
                                    const std::string& label, auto&& cpu_write,
                                    auto&& cpu_pretransform, auto&& gpu_pretransform) {
                constexpr bool has_cpu_pt =
                    !std::is_null_pointer_v<std::decay_t<decltype(cpu_pretransform)>>;
                constexpr bool has_gpu_pt =
                    !std::is_null_pointer_v<std::decay_t<decltype(gpu_pretransform)>>;

                std::visit(
                    [&](auto&& src) {
                        using T = std::decay_t<decltype(src)>;
                        if constexpr (std::is_same_v<T, std::monostate>) {
                            if (dst)
                                cmd->keep_until_pool_reset(std::move(dst));
                        } else if constexpr (std::is_same_v<T, Mesh::DeviceLocal>) {
                            if constexpr (has_gpu_pt) {
                                if (pretransform_mesh) {
                                    if (dst && dst == src.data)
                                        dst.reset();
                                    ensure_buffer(dst, size, label);
                                    gpu_pretransform(src.data, dst, 0);
                                    return;
                                }
                            }
                            if (dst && dst != src.data)
                                cmd->keep_until_pool_reset(std::move(dst));
                            dst = src.data;
                        } else if constexpr (std::is_same_v<T, Mesh::DeviceStaged>) {
                            ensure_buffer(dst, size, label);
                            if constexpr (has_gpu_pt) {
                                if (pretransform_mesh) {
                                    gpu_pretransform(src.data, dst, src.offset);
                                    return;
                                }
                            }
                            cmd->copy(src.data, dst, {vk::BufferCopy{src.offset, 0, size}});
                        } else {
                            ensure_buffer(dst, size, label);
                            const MemoryAllocationHandle alloc = staging->cmd_to_device(cmd, dst);
                            std::function<void(void*)> fill;
                            if constexpr (has_cpu_pt) {
                                if (pretransform_mesh)
                                    fill = [src, cpu_pretransform](void* m) {
                                        cpu_pretransform(src, m);
                                    };
                                else
                                    fill = [src, cpu_write](void* m) { cpu_write(src, m); };
                            } else {
                                fill = [src, cpu_write](void* m) { cpu_write(src, m); };
                            }
                            thread_pool->submit<void>([alloc, fill = std::move(fill)]() {
                                void* mapped = alloc->map();
                                fill(mapped);
                                alloc->unmap();
                            });
                        }
                    },
                    std::forward<decltype(variant)>(variant));
            };

            auto& vb = vertex_buffers[mesh_id];
            auto& prev_vb = prev_vertex_buffers[mesh_id];
            auto& ib = index_buffers[mesh_id];

            if (mesh.vertices_dirty) {
                upload(
                    mesh.get_vertices(), vb, vb_size,
                    fmt::format("Scene::vb[{}]: name={}", mesh_id, mesh.name),
                    [vb_size](auto&& src, void* mapped) {
                        auto* dst = static_cast<PackedVertexData*>(mapped);
                        using S = std::decay_t<decltype(src)>;
                        if constexpr (std::is_same_v<S, Mesh::HostPacked<PackedVertexData>>)
                            std::memcpy(dst, src.data, vb_size);
                        else {
                            static_assert(std::is_same_v<S, Mesh::HostVertices>);
                            src->write(dst);
                        }
                    },
                    [vertex_count, global_transform,
                     global_inverse_transposed_transform](auto&& src, void* mapped) {
                        auto* dst = static_cast<PackedVertexData*>(mapped);
                        using S = std::decay_t<decltype(src)>;
                        if constexpr (std::is_same_v<S, Mesh::HostPacked<PackedVertexData>>) {
                            for (uint32_t v = 0; v < vertex_count; v++)
                                dst[v] =
                                    pretransform_packed_vertex(src.data[v], global_transform,
                                                               global_inverse_transposed_transform);
                        } else {
                            static_assert(std::is_same_v<S, Mesh::HostVertices>);
                            src->write_pretransformed(global_transform,
                                                      global_inverse_transposed_transform, dst);
                        }
                    },
                    [&](const BufferHandle& src_buf, const BufferHandle& dst_buf,
                        vk::DeviceSize src_offset) {
                        pretransform_vertices_gpu(cmd, src_buf, dst_buf, global_transform,
                                                  global_inverse_transposed_transform, vertex_count,
                                                  src_offset);
                    });

                const float4x4 prev_transform =
                    scene_graph[node_id]->prev_global_transform.value_or(global_transform);
                const float4x4 prev_inverse_transposed =
                    scene_graph[node_id]->prev_global_inverse_transposed.value_or(
                        global_inverse_transposed_transform);

                upload(
                    mesh.get_prev_vertices(), prev_vb, prev_vb_size,
                    fmt::format("Scene::prev_vb[{}]: name={}", mesh_id, mesh.name),
                    [prev_vb_size](auto&& src, void* mapped) {
                        auto* dst = static_cast<PackedPrevVertexData*>(mapped);
                        using S = std::decay_t<decltype(src)>;
                        if constexpr (std::is_same_v<S, Mesh::HostPacked<PackedPrevVertexData>>)
                            std::memcpy(dst, src.data, prev_vb_size);
                        else {
                            static_assert(std::is_same_v<S, Mesh::HostPrevVertices>);
                            src->write(dst);
                        }
                    },
                    [vertex_count, prev_transform, prev_inverse_transposed](auto&& src,
                                                                            void* mapped) {
                        auto* dst = static_cast<PackedPrevVertexData*>(mapped);
                        using S = std::decay_t<decltype(src)>;
                        if constexpr (std::is_same_v<S, Mesh::HostPacked<PackedPrevVertexData>>) {
                            for (uint32_t v = 0; v < vertex_count; v++)
                                dst[v] =
                                    pretransform_packed_prev_vertex(src.data[v], prev_transform);
                        } else {
                            static_assert(std::is_same_v<S, Mesh::HostPrevVertices>);
                            src->write_pretransformed(prev_transform, prev_inverse_transposed, dst);
                        }
                    },
                    [&](const BufferHandle& src_buf, const BufferHandle& dst_buf,
                        vk::DeviceSize src_offset) {
                        pretransform_prev_vertices_gpu(cmd, src_buf, dst_buf, prev_transform,
                                                       vertex_count, src_offset);
                    });

                mesh.vertices_dirty = false;
            }

            if (mesh.indices_dirty) {
                upload(
                    mesh.get_indices(), ib, ib_size,
                    fmt::format("Scene::ib[{}]: name={}", mesh_id, mesh.name),
                    [ib_size](auto&& src, void* mapped) {
                        using S = std::decay_t<decltype(src)>;
                        if constexpr (std::is_same_v<S, Mesh::HostPacked<void>>)
                            std::memcpy(mapped, src.data, ib_size);
                        else {
                            static_assert(std::is_same_v<S, Mesh::HostIndices>);
                            src->write(mapped);
                        }
                    },
                    nullptr, nullptr);

                mesh.indices_dirty = false;
            }
        }
    }

    {
        MERIAN_PROFILE_SCOPE("Scene::upload_meshes::wait_idle");
        thread_pool->wait_idle();
    }
}

void Scene::build_blas(const CommandBufferHandle& cmd) {
    MERIAN_PROFILE_SCOPE_GPU(cmd, "Scene::build_blas");
    //     if (mesh_groups.empty())
    //         return;

    blas_geometries.assign(mesh_groups.size(), {});

    bool did_build_static = false;

    for (MeshGroupID group_id = 0; group_id < mesh_groups.size(); group_id++) {
        MeshGroup& group = mesh_groups[group_id];
        if (group.blas && !group.blas_dirty) {
            continue;
        }
        tlas_dirty = true;

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

        const bool static_geometry = !group.has_morphed_mesh && !group.has_animated_node;
        vk::BuildAccelerationStructureFlagsKHR flags;
        if (static_geometry) {
            did_build_static = true;
            flags |= vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
        } else {
            flags |= vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
        }

        if (!group.cached_blas_size_info) {
            group.cached_blas_size_info =
                as_builder.get_size_info(blas_geometry.geometries, blas_geometry.ranges, flags);
        }
        const auto& size_info = *group.cached_blas_size_info;

        if (!group.blas || group.blas->get_size() < size_info.accelerationStructureSize) {
            // cannot reuse and needs to be allcated
            group.blas = allocator->create_acceleration_structure(
                vk::AccelerationStructureTypeKHR::eBottomLevel, size_info,
                fmt::format("Scene::blas[{}]", group_id));
        }

        as_builder.queue_build(blas_geometry.geometries, blas_geometry.ranges, group.blas,
                               size_info, flags);
        group.blas_dirty = false;
        group.blas_last_built_frame = current_frame;
        frame_stats.blas_builds++;
        if (static_geometry)
            frame_stats.blas_builds_static++;
        else
            frame_stats.blas_builds_dynamic++;
    }

    as_builder.get_cmds_blas(cmd, as_scratch_buffer);

    if (did_build_static) {
        // release a probably large scratch buffer
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

        for (NodeID node_id : group.get_instances(meshes)) {
            vk::AccelerationStructureInstanceKHR tlas_instance{};

            tlas_instance.instanceCustomIndex = instance_id;
            tlas_instance.mask = 0xFF;
            tlas_instance.accelerationStructureReference =
                group.blas->get_acceleration_structure_device_address();

            if (group.is_pretranformed(meshes, pretransform_animated)) {
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

    // Advance prev transforms after upload_meshes (prev vertex pretransform reads them).
    for (const NodeID& nid : node_ids) {
        assert(scene_graph[nid]);
        SceneNode& n = *scene_graph[nid];
        n.prev_global_transform = n.global_transform;
        n.prev_global_inverse_transposed = n.global_inverse_transposed;
    }

    // changes at regroup, or updated mesh (buffer address) -> run every frame.
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
