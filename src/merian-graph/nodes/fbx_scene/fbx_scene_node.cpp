#include "merian-graph/nodes/fbx_scene/fbx_scene_node.hpp"

#include "merian/vk/command/queue.hpp"

#include <spdlog/spdlog.h>

namespace merian {

FBXSceneNode::FBXSceneNode() : Node() {}

DeviceSupportInfo
FBXSceneNode::query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
#ifndef MERIAN_UFBX_ENABLED
    return DeviceSupportInfo{false, "FBX support disabled at compile time (ufbx option)"};
#else
    return DeviceSupportInfo::check(query_info, {"accelerationStructure", "scalarBlockLayout",
                                                 "shaderInt64", "bufferDeviceAddress"});
#endif
}

void FBXSceneNode::initialize([[maybe_unused]] const ContextHandle& context,
                              [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
#ifdef MERIAN_UFBX_ENABLED
    this->context = context;
    this->allocator = allocator;
    compile_context = context->get_shader_compile_context();
    texture_manager = std::make_shared<TextureManager>(compile_context, context, allocator, 4096);
    material_system =
        std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);
    scene = std::make_shared<FBXScene>(compile_context, context, allocator, material_system);
#endif
}

std::vector<InputConnectorDescriptor> FBXSceneNode::describe_inputs() {
    return {{"controller", con_controller}};
}

std::vector<OutputConnectorDescriptor>
FBXSceneNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    return {{"scene", con_scene}};
}

void FBXSceneNode::process([[maybe_unused]] GraphRun& run,
                           [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                           [[maybe_unused]] const NodeIO& io) {
#ifdef MERIAN_UFBX_ENABLED
    const auto& cmd = run.get_cmd();

    if (io.is_connected(con_controller)) {
        const InputControllerHandle& input = io[con_controller];
        if (input && input != registered_controller.lock()) {
            input->add_listener(cam_controller);
            registered_controller = input;
        }
        cam_controller->attach(scene->get_active_camera());
        cam_controller->update(run.get_time_delta());
    }

    scene->update(cmd, static_cast<float>(run.get_elapsed()),
                  static_cast<float>(run.get_time_delta()), run.get_total_iteration());

    const Scene::UpdateChanges& changes = scene->get_last_update_changes();
    if (changes.geometry_changed)
        io.send_event("geometry_changed");
    if (changes.transform_changed)
        io.send_event("transform_changed");
    if (changes.camera_changed)
        io.send_event("camera_changed");

    io[con_scene] = std::static_pointer_cast<Scene>(scene);
#endif
}

FBXSceneNode::NodeStatusFlags FBXSceneNode::properties([[maybe_unused]] Properties& config) {
#ifdef MERIAN_UFBX_ENABLED
    std::string path_str = file_path.string();
    if (config.config_text("file", path_str, true, "Path to .fbx file")) {
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
