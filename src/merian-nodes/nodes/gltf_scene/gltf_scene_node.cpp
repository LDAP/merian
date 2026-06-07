#include "merian-nodes/nodes/gltf_scene/gltf_scene_node.hpp"

#include <spdlog/spdlog.h>

namespace merian {

GLTFSceneNode::GLTFSceneNode() : Node() {}

DeviceSupportInfo
GLTFSceneNode::query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
#ifndef MERIAN_TINYGLTF_ENABLED
    return DeviceSupportInfo{false, "GLTF support disabled at compile time (tinygltf option)"};
#else
    return DeviceSupportInfo::check(query_info, {"accelerationStructure", "scalarBlockLayout"});
#endif
}

void GLTFSceneNode::initialize([[maybe_unused]] const ContextHandle& context,
                               [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
#ifdef MERIAN_TINYGLTF_ENABLED
    this->context = context;
    this->allocator = allocator;
    compile_context = context->get_shader_compile_context();
    texture_manager = std::make_shared<TextureManager>(compile_context, context, allocator, 4096);
    material_system =
        std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);
    scene = std::make_shared<GLTFScene>(compile_context, context, allocator, material_system);
#endif
}

std::vector<OutputConnectorDescriptor>
GLTFSceneNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    return {{"scene", con_scene}};
}

void GLTFSceneNode::process([[maybe_unused]] GraphRun& run,
                            [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                            [[maybe_unused]] const NodeIO& io) {
#ifdef MERIAN_TINYGLTF_ENABLED
    const auto& cmd = run.get_cmd();

    if (needs_load && !file_path.empty()) {
        scene->load(cmd, file_path);
        needs_load = false;
    }

    scene->update(cmd, static_cast<float>(run.get_elapsed()),
                  static_cast<float>(run.get_time_delta()), run.get_total_iteration());

    io[con_scene] = std::static_pointer_cast<Scene>(scene);
#endif
}

GLTFSceneNode::NodeStatusFlags GLTFSceneNode::properties([[maybe_unused]] Properties& config) {
#ifdef MERIAN_TINYGLTF_ENABLED
    std::string path_str = file_path.string();
    if (config.config_text("file", path_str, true, "Path to .gltf or .glb file")) {
        file_path = path_str;
        needs_load = true;
    }

    if (scene) {
        scene->properties(config);
    }
#endif

    return {};
}

} // namespace merian
