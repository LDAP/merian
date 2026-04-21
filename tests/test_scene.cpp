#include <gtest/gtest.h>

#include "merian-shaders/scene/scene.hpp"
#include "merian-shaders/shading/materials/material_system.hpp"
#include "merian-shaders/utils/texture_manager.hpp"
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

// Concrete Scene subclass that exposes the protected add_* methods for testing.
class TestScene : public Scene {
  public:
    using Scene::Scene;

    using Scene::add_camera;
    using Scene::add_mesh;
    using Scene::add_mesh_instance;
    using Scene::add_node;
};

class SceneTest : public ::testing::Test {
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
            .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
            .application_name = "test-scene",
        };
        context = Context::create(info);
        auto resources = context->get_context_extension<ExtensionResources>();
        allocator = resources->resource_allocator();
        queue = context->get_queue_GCT();
        compile_context = ShaderCompileContext::create(context);
        compile_context->add_search_path(TEST_SHADER_DIR);
        obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);
        texture_manager = std::make_shared<TextureManager>(compile_context, context, allocator,
                                                           obj_allocator, 16);
        material_system = std::make_shared<MaterialSystem>(compile_context, context, allocator,
                                                           obj_allocator, texture_manager);
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

    // Helper: create a triangle mesh
    static MeshHandle make_triangle(MaterialID mat_id,
                                    GeometryFlags flags = GeometryFlags::ForceOpaque) {
        std::unique_ptr<SimpleMesh> m = std::make_unique<SimpleMesh>();
        m->material_id = mat_id;
        m->flags = flags;

        PackedVertexData v0{}, v1{}, v2{};
        v0.position = float3(0, 0, 0);
        v1.position = float3(1, 0, 0);
        v2.position = float3(0, 1, 0);
        m->vertices = {v0, v1, v2};
        m->indices = {uint3(0, 1, 2)};
        return m;
    }
};

ContextHandle SceneTest::context;
ResourceAllocatorHandle SceneTest::allocator;
QueueHandle SceneTest::queue;
ShaderCompileContextHandle SceneTest::compile_context;
ShaderObjectAllocatorHandle SceneTest::obj_allocator;
TextureManagerHandle SceneTest::texture_manager;
MaterialSystemHandle SceneTest::material_system;

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

TEST_F(SceneTest, Construction) {
    auto scene = std::make_shared<TestScene>(compile_context, context, allocator, obj_allocator,
                                             material_system);
    EXPECT_NE(scene->get_composition(), nullptr);
    EXPECT_EQ(scene->get_material_system(), material_system);
    EXPECT_EQ(scene->get_active_camera(), nullptr);
}

// ---------------------------------------------------------------------------
// Scene graph: add nodes and compute world transforms
// ---------------------------------------------------------------------------

TEST_F(SceneTest, SceneGraphTransforms) {
    auto scene = std::make_shared<TestScene>(compile_context, context, allocator, obj_allocator,
                                             material_system);

    SceneNode root;
    root.name = "root";
    root.local_transform = translation(float3(2, 0, 0));
    NodeID root_id = scene->add_node(root);
    EXPECT_EQ(root_id, 0u);

    SceneNode child;
    child.name = "child";
    child.parent = root_id;
    child.local_transform = translation(float3(0, 3, 0));
    NodeID child_id = scene->add_node(child);
    EXPECT_EQ(child_id, 1u);

    const auto& graph = scene->get_scene_graph();
    EXPECT_FLOAT_EQ(graph[root_id].global_transform.value()[0][3], 2.0f);
    EXPECT_FLOAT_EQ(graph[root_id].global_transform.value()[1][3], 0.0f);
    // Child: global = mul(root, child) => translate (2, 3, 0)
    EXPECT_FLOAT_EQ(graph[child_id].global_transform.value()[0][3], 2.0f);
    EXPECT_FLOAT_EQ(graph[child_id].global_transform.value()[1][3], 3.0f);
}

// ---------------------------------------------------------------------------
// Mesh grouping: static non-instanced meshes go into one group
// ---------------------------------------------------------------------------

TEST_F(SceneTest, MeshGroupingStatic) {
    auto scene = std::make_shared<TestScene>(compile_context, context, allocator, obj_allocator,
                                             material_system);

    SceneNode node;
    node.name = "node0";
    NodeID nid = scene->add_node(node);

    MeshID m0 = scene->add_mesh(make_triangle(0));
    MeshID m1 = scene->add_mesh(make_triangle(0));
    scene->add_mesh_instance(m0, nid);
    scene->add_mesh_instance(m1, nid);

    // Trigger update which calls create_mesh_groups + upload
    queue->submit_wait([&](const CommandBufferHandle& cmd) { scene->update(cmd, 0.0f, 0.0f, 0); });

    // Both meshes are static non-instanced → single group with both meshes
    EXPECT_EQ(scene->get_mesh(m0).instances.size(), 1u);
    EXPECT_EQ(scene->get_mesh(m1).instances.size(), 1u);
}

// ---------------------------------------------------------------------------
// Mesh grouping: instanced meshes share the same instance set
// ---------------------------------------------------------------------------

TEST_F(SceneTest, MeshGroupingInstanced) {
    auto scene = std::make_shared<TestScene>(compile_context, context, allocator, obj_allocator,
                                             material_system);

    SceneNode n0, n1;
    n0.name = "inst0";
    n1.name = "inst1";
    NodeID nid0 = scene->add_node(n0);
    NodeID nid1 = scene->add_node(n1);

    MeshID mid = scene->add_mesh(make_triangle(0));
    scene->add_mesh_instance(mid, nid0);
    scene->add_mesh_instance(mid, nid1);

    EXPECT_EQ(scene->get_mesh(mid).instances.size(), 2u);

    queue->submit_wait([&](const CommandBufferHandle& cmd) { scene->update(cmd, 0.0f, 0.0f, 0); });
}

// ---------------------------------------------------------------------------
// Mesh grouping: dynamic non-instanced meshes grouped by node
// ---------------------------------------------------------------------------

TEST_F(SceneTest, MeshGroupingDynamic) {
    auto scene = std::make_shared<TestScene>(compile_context, context, allocator, obj_allocator,
                                             material_system);

    SceneNode n0, n1;
    n0.name = "dyn0";
    n1.name = "dyn1";
    NodeID nid0 = scene->add_node(n0);
    NodeID nid1 = scene->add_node(n1);

    // Two dynamic meshes on different nodes
    MeshID m0 =
        scene->add_mesh(make_triangle(0, GeometryFlags::ForceOpaque | GeometryFlags::IsDynamic));
    MeshID m1 =
        scene->add_mesh(make_triangle(0, GeometryFlags::ForceOpaque | GeometryFlags::IsDynamic));
    scene->add_mesh_instance(m0, nid0);
    scene->add_mesh_instance(m1, nid1);

    queue->submit_wait([&](const CommandBufferHandle& cmd) { scene->update(cmd, 0.0f, 0.0f, 0); });

    // Each dynamic mesh on a different node → separate groups
    EXPECT_EQ(scene->get_mesh(m0).instances.size(), 1u);
    EXPECT_EQ(scene->get_mesh(m1).instances.size(), 1u);
}

// ---------------------------------------------------------------------------
// Camera management
// ---------------------------------------------------------------------------

TEST_F(SceneTest, CameraManagement) {
    auto scene = std::make_shared<TestScene>(compile_context, context, allocator, obj_allocator,
                                             material_system);

    EXPECT_EQ(scene->get_active_camera(), nullptr);

    auto cam = std::make_shared<Camera>();
    scene->add_camera(cam);
    EXPECT_EQ(scene->get_active_camera(), cam);

    auto cam2 = std::make_shared<Camera>();
    scene->add_camera(cam2);
    scene->set_active_camera(1);
    EXPECT_EQ(scene->get_active_camera(), cam2);
}
