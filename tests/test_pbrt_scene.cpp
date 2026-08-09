#include <gtest/gtest.h>

#include "merian-shaders/scene/pbrt_scene.hpp"
#include "merian-shaders/shading/materials/material_system.hpp"
#include "merian-shaders/utils/texture_manager.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"

#include <fstream>

using namespace merian;

#ifndef TEST_SHADER_DIR
#define TEST_SHADER_DIR "."
#endif

namespace {

// Quad floor with a named diffuse material and an emissive sphere.
constexpr const char* TEST_SCENE = R"(
Integrator "path"
LookAt 0 1 5  0 1 0  0 1 0
Camera "perspective" "float fov" 40
Film "rgb" "integer xresolution" 64 "integer yresolution" 64
WorldBegin
MakeNamedMaterial "White" "string type" "diffuse" "rgb reflectance" [0.7 0.7 0.7]
NamedMaterial "White"
Shape "trianglemesh" "integer indices" [0 1 2 0 2 3]
    "point3 P" [-1 0 -1  -1 0 1  1 0 1  1 0 -1]
AttributeBegin
    AreaLightSource "diffuse" "rgb L" [10 10 10]
    Translate 0 2 0
    Shape "sphere" "float radius" 0.25
AttributeEnd
)";

} // namespace

class PBRTSceneTest : public ::testing::Test {
  protected:
    static ContextHandle context;
    static ResourceAllocatorHandle allocator;
    static QueueHandle queue;
    static ShaderCompileContextHandle compile_context;
    static TextureManagerHandle texture_manager;
    static MaterialSystemHandle material_system;

    static void SetUpTestSuite() {
        spdlog::set_level(spdlog::level::debug);
        ContextCreateInfo info{
            .features =
                VulkanFeatures({"scalarBlockLayout", "shaderInt64", "shaderFloat16",
                                "accelerationStructure", "storageBuffer16BitAccess",
                                "storageBuffer8BitAccess", "uniformAndStorageBuffer8BitAccess"}),
            .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
            .application_name = "test-pbrt-scene",
        };
        context = Context::create(info);
        auto resources = context->get_context_extension<ExtensionResources>();
        allocator = resources->resource_allocator();
        queue = context->get_queue_GCT();
        compile_context = ShaderCompileContext::create(context);
        compile_context->add_search_path(TEST_SHADER_DIR);
        texture_manager = std::make_shared<TextureManager>(compile_context, context, allocator, 16);
        material_system =
            std::make_shared<MaterialSystem>(compile_context, context, allocator, texture_manager);
    }

    static void TearDownTestSuite() {
        context->get_device()->get_device().waitIdle();
        material_system.reset();
        texture_manager.reset();
        compile_context.reset();
        allocator.reset();
        queue.reset();
        context.reset();
    }
};

ContextHandle PBRTSceneTest::context;
ResourceAllocatorHandle PBRTSceneTest::allocator;
QueueHandle PBRTSceneTest::queue;
ShaderCompileContextHandle PBRTSceneTest::compile_context;
TextureManagerHandle PBRTSceneTest::texture_manager;
MaterialSystemHandle PBRTSceneTest::material_system;

TEST_F(PBRTSceneTest, LoadMiniScene) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "merian-test-scene.pbrt";
    {
        std::ofstream file(path);
        file << TEST_SCENE;
    }

    auto scene = std::make_shared<PBRTScene>(compile_context, context, allocator, material_system);
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        scene->load(cmd, path);
        scene->update(cmd, 0.0f, 0.0f, 0);
    });
    std::filesystem::remove(path);

    ASSERT_TRUE(scene->is_ready());
    // root + one node per shape
    EXPECT_GE(scene->get_scene_graph().size(), 3u);
    // the white quad and the emissive sphere copy
    EXPECT_EQ(scene->get_material_system()->get_material_count(), 2u);
    EXPECT_EQ(scene->get_cameras().size(), 1u);
    EXPECT_TRUE(std::as_const(*scene).get_aabb().is_valid());
}

TEST_F(PBRTSceneTest, MissingFileReportsNotReady) {
    auto scene = std::make_shared<PBRTScene>(compile_context, context, allocator, material_system);
    queue->submit_wait(
        [&](const CommandBufferHandle& cmd) { scene->load(cmd, "does-not-exist.pbrt"); });
    EXPECT_FALSE(scene->is_ready());
}
