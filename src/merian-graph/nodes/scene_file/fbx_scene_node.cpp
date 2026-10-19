#include "merian-graph/nodes/scene_file/fbx_scene_node.hpp"

#ifdef MERIAN_UFBX_ENABLED
#include "merian-shaders/scene/fbx_scene.hpp"
#endif

namespace merian {

FBXSceneNode::FBXSceneNode() : SceneFileNode("Path to .fbx file") {}

DeviceSupportInfo
FBXSceneNode::query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
#ifdef MERIAN_UFBX_ENABLED
    return SceneFileNode::query_device_support(query_info);
#else
    return DeviceSupportInfo{false, "FBX support disabled at compile time (ufbx option)"};
#endif
}

SceneHandle
FBXSceneNode::create_scene([[maybe_unused]] const ShaderCompileContextHandle& compile_context,
                           [[maybe_unused]] const ContextHandle& context,
                           [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                           [[maybe_unused]] const MaterialSystemHandle& material_system) {
#ifdef MERIAN_UFBX_ENABLED
    return std::make_shared<FBXScene>(compile_context, context, allocator, material_system);
#else
    return nullptr;
#endif
}

void FBXSceneNode::load_file([[maybe_unused]] Scene& scene,
                             [[maybe_unused]] const CommandBufferHandle& cmd,
                             [[maybe_unused]] const std::filesystem::path& path) {
#ifdef MERIAN_UFBX_ENABLED
    static_cast<FBXScene&>(scene).load(cmd, path);
#endif
}

} // namespace merian
