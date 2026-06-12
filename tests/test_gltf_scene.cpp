#include <gtest/gtest.h>

#include "merian-scene/gltf_scene.hpp"
#include "merian-scene/material_system.hpp"
#include "merian-scene/texture_manager.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"

using namespace merian;

#ifndef TEST_SHADER_DIR
#define TEST_SHADER_DIR "."
#endif

#ifndef TEST_MODEL_DIR
#define TEST_MODEL_DIR "."
#endif

class GLTFSceneTest : public ::testing::Test {
  protected:
    static ContextHandle context;
    static ResourceAllocatorHandle allocator;
    static QueueHandle queue;
    static ShaderCompileContextHandle compile_context;
    static ShaderObjectAllocatorHandle obj_allocator;
    static TextureManagerHandle texture_manager;
    static MaterialSystemHandle material_system;

    static void SetUpTestSuite() {
        spdlog::set_level(spdlog::level::debug);
        ContextCreateInfo info{
            .features = VulkanFeatures({"scalarBlockLayout", "shaderInt64"}),
            .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
            .application_name = "test-gltf-scene",
        };
        context = Context::create(info);
        auto resources = context->get_context_extension<ExtensionResources>();
        allocator = resources->resource_allocator();
        queue = context->get_queue_GCT();
        compile_context = ShaderCompileContext::create(context);
        compile_context->add_search_path(TEST_SHADER_DIR);
        obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);
        texture_manager =
            std::make_shared<TextureManager>(compile_context, context, allocator, 16);
        material_system =
            std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);
    }

    static void TearDownTestSuite() {
        context->get_device()->get_device().waitIdle();
        material_system.reset();
        texture_manager.reset();
        obj_allocator.reset();
        compile_context.reset();
        allocator.reset();
        queue.reset();
        context.reset();
    }
};

ContextHandle GLTFSceneTest::context;
ResourceAllocatorHandle GLTFSceneTest::allocator;
QueueHandle GLTFSceneTest::queue;
ShaderCompileContextHandle GLTFSceneTest::compile_context;
ShaderObjectAllocatorHandle GLTFSceneTest::obj_allocator;
TextureManagerHandle GLTFSceneTest::texture_manager;
MaterialSystemHandle GLTFSceneTest::material_system;

// ---------------------------------------------------------------------------
// Load the Cube.gltf test model
// ---------------------------------------------------------------------------

TEST_F(GLTFSceneTest, LoadCubeGLTF) {
    auto scene = std::make_shared<GLTFScene>(compile_context, context, allocator,
                                             material_system);

    std::filesystem::path cube_path = std::filesystem::path(TEST_MODEL_DIR) / "Cube" / "Cube.gltf";
    ASSERT_TRUE(std::filesystem::exists(cube_path)) << "Test model not found: " << cube_path;

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        scene->load(cmd, cube_path);
        scene->update(cmd, 0.0f, 0.0f, 0);
    });

    // Cube.gltf has 1 mesh with 1 primitive, 1 material, and 1 node
    const auto& graph = scene->get_scene_graph();
    EXPECT_GE(graph.size(), 1u);
    EXPECT_EQ(scene->get_material_system()->get_material_count(), 1u);
}

// ---------------------------------------------------------------------------
// Load the box01.glb test model
// ---------------------------------------------------------------------------

TEST_F(GLTFSceneTest, LoadBoxGLB) {
    // Reset material system for a clean state
    texture_manager =
        std::make_shared<TextureManager>(compile_context, context, allocator, 16);
    material_system =
        std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);

    auto scene = std::make_shared<GLTFScene>(compile_context, context, allocator,
                                             material_system);

    std::filesystem::path glb_path = std::filesystem::path(TEST_MODEL_DIR) / "box01.glb";
    ASSERT_TRUE(std::filesystem::exists(glb_path)) << "Test model not found: " << glb_path;

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        scene->load(cmd, glb_path);
        scene->update(cmd, 0.0f, 0.0f, 0);
    });

    const auto& graph = scene->get_scene_graph();
    EXPECT_GE(graph.size(), 1u);
}

// ---------------------------------------------------------------------------
// Scene graph hierarchy is preserved
// ---------------------------------------------------------------------------

TEST_F(GLTFSceneTest, SceneGraphHierarchy) {
    texture_manager =
        std::make_shared<TextureManager>(compile_context, context, allocator, 16);
    material_system =
        std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);

    auto scene = std::make_shared<GLTFScene>(compile_context, context, allocator,
                                             material_system);

    std::filesystem::path cube_path = std::filesystem::path(TEST_MODEL_DIR) / "Cube" / "Cube.gltf";
    ASSERT_TRUE(std::filesystem::exists(cube_path));

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        scene->load(cmd, cube_path);
        scene->update(cmd, 0.0f, 0.0f, 0);
    });

    // All root nodes should have parent == NODE_ID_INVALID
    const auto& graph = scene->get_scene_graph();
    bool has_root = false;
    for (const auto& node : graph) {
        if (node && node->parent == Scene::NODE_ID_INVALID) {
            has_root = true;
            break;
        }
    }
    EXPECT_TRUE(has_root);
}
