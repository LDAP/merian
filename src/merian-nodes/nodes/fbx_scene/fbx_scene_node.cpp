#include "merian-nodes/nodes/fbx_scene/fbx_scene_node.hpp"

#include <spdlog/spdlog.h>

namespace merian {

FBXSceneNode::FBXSceneNode() : Node() {}

DeviceSupportInfo
FBXSceneNode::query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
#ifndef MERIAN_UFBX_ENABLED
    return DeviceSupportInfo{false, "FBX support disabled at compile time (ufbx option)"};
#else
    return DeviceSupportInfo::check(query_info, {"accelerationStructure", "scalarBlockLayout"});
#endif
}

void FBXSceneNode::initialize([[maybe_unused]] const ContextHandle& context,
                              [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
#ifdef MERIAN_UFBX_ENABLED
    this->context = context;
    this->allocator = allocator;
    compile_context = ShaderCompileContext::create(context);
    texture_manager = std::make_shared<TextureManager>(compile_context, context, allocator, 4096);
    material_system =
        std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);
    scene = std::make_shared<FBXScene>(compile_context, context, allocator, material_system);
#endif
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

    if (needs_load && !file_path.empty()) {
        scene->load(cmd, file_path);
        needs_load = false;
    }

    scene->update(cmd, static_cast<float>(run.get_elapsed()),
                  static_cast<float>(run.get_time_delta()), run.get_total_iteration());

    io[con_scene] = std::static_pointer_cast<Scene>(scene);
#endif
}

FBXSceneNode::NodeStatusFlags FBXSceneNode::properties([[maybe_unused]] Properties& config) {
#ifdef MERIAN_UFBX_ENABLED
    std::string path_str = file_path.string();
    if (config.config_text("file", path_str, true, "Path to .fbx file")) {
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
