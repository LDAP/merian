#include "merian-graph/nodes/scene_file/gltf_scene_node.hpp"

#ifdef MERIAN_TINYGLTF_ENABLED
#include "merian-shaders/scene/gltf_scene.hpp"
#endif

namespace merian {

GLTFSceneNode::GLTFSceneNode() : SceneFileNode("Path to .gltf or .glb file") {}

DeviceSupportInfo
GLTFSceneNode::query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
#ifdef MERIAN_TINYGLTF_ENABLED
    return SceneFileNode::query_device_support(query_info);
#else
    return DeviceSupportInfo{false, "GLTF support disabled at compile time (tinygltf option)"};
#endif
}

SceneHandle
GLTFSceneNode::create_scene([[maybe_unused]] const ShaderCompileContextHandle& compile_context,
                            [[maybe_unused]] const ContextHandle& context,
                            [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                            [[maybe_unused]] const MaterialSystemHandle& material_system) {
#ifdef MERIAN_TINYGLTF_ENABLED
    return std::make_shared<GLTFScene>(compile_context, context, allocator, material_system);
#else
    return nullptr;
#endif
}

void GLTFSceneNode::load_file([[maybe_unused]] Scene& scene,
                              [[maybe_unused]] const CommandBufferHandle& cmd,
                              [[maybe_unused]] const std::filesystem::path& path) {
#ifdef MERIAN_TINYGLTF_ENABLED
    static_cast<GLTFScene&>(scene).load(cmd, path);
#endif
}

} // namespace merian
