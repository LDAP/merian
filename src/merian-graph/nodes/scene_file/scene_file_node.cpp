#include "merian-graph/nodes/scene_file/scene_file_node.hpp"

#include "merian/vk/command/queue.hpp"

namespace merian {

SceneFileNode::SceneFileNode(std::string file_description)
    : file_description(std::move(file_description)) {}

DeviceSupportInfo SceneFileNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return Scene::query_device_support(query_info);
}

void SceneFileNode::initialize(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator) {
    this->context = context;
    const ShaderCompileContextHandle compile_context = context->get_shader_compile_context();
    const TextureManagerHandle texture_manager =
        std::make_shared<TextureManager>(compile_context, context, allocator, 4096);
    const MaterialSystemHandle material_system =
        std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);
    scene = create_scene(compile_context, context, allocator, material_system);
}

const SceneHandle& SceneFileNode::get_scene() const {
    return scene;
}

std::vector<InputConnectorDescriptor> SceneFileNode::describe_inputs() {
    return {{"controller", con_controller, {}, 0, true}};
}

std::vector<OutputConnectorDescriptor>
SceneFileNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    return {{.name = "scene", .connector = con_scene, .persistent = true}};
}

SceneFileNode::NodeStatusFlags
SceneFileNode::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                            const NodeIO& io,
                            [[maybe_unused]] const NodeConnectionInfo& info,
                            [[maybe_unused]] Submission& submission) {
    io[con_scene] = scene;
    return {};
}

SceneFileNode::NodeStatusFlags
SceneFileNode::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    if (scene->is_ready() && io.is_connected(con_controller)) {
        const InputControllerHandle& input = io[con_controller];
        if (input && input != registered_controller.lock()) {
            input->add_listener(cam_controller);
            registered_controller = input;
        }
        cam_controller->attach(scene->get_active_camera());
        cam_controller->update(info.get_time_delta());
    }

    scene->update(submission, static_cast<float>(info.get_elapsed()),
                  static_cast<float>(info.get_time_delta()),
                  static_cast<uint32_t>(info.get_iteration()), io.get_seed(),
                  info.get_shader_object_allocator());

    const uint32_t changes = scene->get_last_updates();
    if ((changes & Scene::SceneUpdateGeometry) != 0)
        io.send_event("geometry_changed");
    if ((changes & Scene::SceneUpdateTransform) != 0)
        io.send_event("transform_changed");
    if ((changes & Scene::SceneUpdateCamera) != 0)
        io.send_event("camera_changed");

    return {};
}

SceneFileNode::NodeStatusFlags SceneFileNode::properties(Properties& config) {
    if (config.config_path("file", file_path, file_description)) {
        // Load synchronously so the stored camera is restored onto a populated scene.
        if (scene && !file_path.empty()) {
            context->get_queue(vk::QueueFlagBits::eGraphics)
                ->submit_wait(
                    [&](const CommandBufferHandle& cmd) { load_file(*scene, cmd, file_path); });
        }
    }

    if (scene) {
        scene->properties(config);
    }

    if (config.st_begin_child("camera controller", "Camera Controller")) {
        cam_controller->properties(config);
        config.st_end_child();
    }

    return {};
}

} // namespace merian
