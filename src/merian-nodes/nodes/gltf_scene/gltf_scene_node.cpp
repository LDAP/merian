#include "merian-nodes/nodes/gltf_scene/gltf_scene_node.hpp"

#include <spdlog/spdlog.h>

namespace merian {

GLTFSceneNode::GLTFSceneNode() : Node() {}

DeviceSupportInfo GLTFSceneNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(query_info,
                                    {"accelerationStructure", "scalarBlockLayout"});
}

void GLTFSceneNode::initialize(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    compile_context = ShaderCompileContext::create(context);
    obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);
    texture_manager =
        std::make_shared<TextureManager>(compile_context, context, allocator, obj_allocator, 4096);
    material_system = std::make_shared<MaterialSystem>(compile_context, context, allocator,
                                                       obj_allocator, texture_manager);

    // Create an empty scene so there's always a valid output
    scene = std::make_shared<GLTFScene>(compile_context, context, allocator, obj_allocator,
                                        material_system);
}

std::vector<OutputConnectorDescriptor>
GLTFSceneNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    return {{"scene", con_scene}};
}

void GLTFSceneNode::process(GraphRun& run,
                            [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                            const NodeIO& io) {
    const auto& cmd = run.get_cmd();

    if (needs_load && !file_path.empty()) {
        scene = std::make_shared<GLTFScene>(compile_context, context, allocator, obj_allocator,
                                             material_system);
        scene->load(cmd, file_path);
        needs_load = false;
        frame = 0;
    }

    scene->update(cmd, static_cast<float>(run.get_elapsed()),
                  static_cast<float>(run.get_time_delta()), frame++);

    io[con_scene] = std::static_pointer_cast<Scene>(scene);
}

GLTFSceneNode::NodeStatusFlags GLTFSceneNode::properties(Properties& config) {
    std::string path_str = file_path.string();
    if (config.config_text("file", path_str, true, "Path to .gltf or .glb file")) {
        file_path = path_str;
        needs_load = true;
    }

    if (scene) {
        bool pretransform = scene->get_pretransform_dynamic();
        if (config.config_bool("pretransform dynamic", pretransform,
                               "Bake dynamic mesh transforms on CPU (debug / small scenes)")) {
            scene->set_pretransform_dynamic(pretransform);
        }
        config.output_text("nodes: {}, meshes: {}, materials: {}",
                           scene->get_scene_graph().size(), 0 /* no public mesh count */, 0);
    }

    return {};
}

} // namespace merian
