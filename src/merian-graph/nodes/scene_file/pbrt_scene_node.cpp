#include "merian-graph/nodes/scene_file/pbrt_scene_node.hpp"

#ifdef MERIAN_PBRT_ENABLED
#include "merian-shaders/scene/pbrt_scene.hpp"
#endif

namespace merian {

PBRTSceneNode::PBRTSceneNode() : SceneFileNode("Path to .pbrt file") {}

DeviceSupportInfo
PBRTSceneNode::query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
#ifdef MERIAN_PBRT_ENABLED
    return SceneFileNode::query_device_support(query_info);
#else
    return DeviceSupportInfo{false, "pbrt support disabled at compile time (pbrt option)"};
#endif
}

SceneHandle
PBRTSceneNode::create_scene([[maybe_unused]] const ShaderCompileContextHandle& compile_context,
                            [[maybe_unused]] const ContextHandle& context,
                            [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                            [[maybe_unused]] const MaterialSystemHandle& material_system) {
#ifdef MERIAN_PBRT_ENABLED
    return std::make_shared<PBRTScene>(compile_context, context, allocator, material_system);
#else
    return nullptr;
#endif
}

void PBRTSceneNode::load_file([[maybe_unused]] Scene& scene,
                              [[maybe_unused]] const CommandBufferHandle& cmd,
                              [[maybe_unused]] const std::filesystem::path& path) {
#ifdef MERIAN_PBRT_ENABLED
    static_cast<PBRTScene&>(scene).load(cmd, path);
#endif
}

} // namespace merian
