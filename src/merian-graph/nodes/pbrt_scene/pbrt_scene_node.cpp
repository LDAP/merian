#include "merian-graph/nodes/pbrt_scene/pbrt_scene_node.hpp"

#include "merian/vk/command/queue.hpp"

#include <spdlog/spdlog.h>

namespace merian {

PBRTSceneNode::PBRTSceneNode() : Node() {}

DeviceSupportInfo
PBRTSceneNode::query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
#ifndef MERIAN_PBRT_ENABLED
    return DeviceSupportInfo{false, "pbrt support disabled at compile time (pbrt option)"};
#else
    return Scene::query_device_support(query_info);
#endif
}

void PBRTSceneNode::initialize([[maybe_unused]] const ContextHandle& context,
                               [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
#ifdef MERIAN_PBRT_ENABLED
    this->context = context;
    this->allocator = allocator;
    compile_context = context->get_shader_compile_context();
    texture_manager = std::make_shared<TextureManager>(compile_context, context, allocator, 4096);
    material_system =
        std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);
    scene = std::make_shared<PBRTScene>(compile_context, context, allocator, material_system);
#endif
}

std::vector<InputConnectorDescriptor> PBRTSceneNode::describe_inputs() {
    return {{"controller", con_controller, {}, 0, true}};
}

std::vector<OutputConnectorDescriptor>
PBRTSceneNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    return {{"scene", con_scene}};
}

[[nodiscard]] PBRTSceneNode::NodeStatusFlags
PBRTSceneNode::process([[maybe_unused]] const NodeIO& io,
                       [[maybe_unused]] const NodeProcessInfo& info,
                       Submission& submission) {
#ifdef MERIAN_PBRT_ENABLED
    const auto& cmd = submission.get_cmd();

    if (scene->is_ready() && io.is_connected(con_controller)) {
        const InputControllerHandle& input = io[con_controller];
        if (input && input != registered_controller.lock()) {
            input->add_listener(cam_controller);
            registered_controller = input;
        }
        cam_controller->attach(scene->get_active_camera());
        cam_controller->update(info.get_time_delta());
    }

    scene->update(cmd, static_cast<float>(info.get_elapsed()),
                  static_cast<float>(info.get_time_delta()), info.get_total_iteration(),
                  info.get_shader_object_allocator());

    const Scene::UpdateChanges& changes = scene->get_last_update_changes();
    if (changes.geometry_changed)
        io.send_event("geometry_changed");
    if (changes.transform_changed)
        io.send_event("transform_changed");
    if (changes.camera_changed)
        io.send_event("camera_changed");

#endif
    return {};
}

PBRTSceneNode::NodeStatusFlags
PBRTSceneNode::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                            [[maybe_unused]] const NodeIO& io,
                            [[maybe_unused]] const NodeConnectionInfo& info,
                            [[maybe_unused]] Submission& submission) {
#ifdef MERIAN_PBRT_ENABLED
    io[con_scene] = std::static_pointer_cast<Scene>(scene);
#endif
    return {};
}

PBRTSceneNode::NodeStatusFlags PBRTSceneNode::properties([[maybe_unused]] Properties& config) {
#ifdef MERIAN_PBRT_ENABLED
    std::string path_str = file_path.string();
    if (config.config_text("file", path_str, true, "Path to .pbrt file")) {
        file_path = path_str;
        // Load synchronously so the stored camera is restored onto a populated scene.
        if (scene && !file_path.empty()) {
            context->get_queue_GCT()->submit_wait(
                [&](const CommandBufferHandle& cmd) { scene->load(cmd, file_path); });
        }
    }

    if (scene) {
        scene->properties(config);
    }

    if (config.st_begin_child("camera_controller", "Camera Controller")) {
        cam_controller->properties(config);
        config.st_end_child();
    }
#endif

    return {};
}

} // namespace merian
