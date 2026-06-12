#include <gtest/gtest.h>

#include "merian-scene/texture_manager.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

#include <bit>
#include <cstring>

using namespace merian;

#ifndef TEST_SHADER_DIR
#define TEST_SHADER_DIR "."
#endif

class TextureManagerTest : public ::testing::Test {
  protected:
    static ContextHandle context;
    static ResourceAllocatorHandle allocator;
    static QueueHandle queue;
    static ShaderCompileContextHandle compile_context;
    static ShaderObjectAllocatorHandle obj_allocator;

    static void SetUpTestSuite() {
        spdlog::set_level(spdlog::level::debug);
        ContextCreateInfo info{
            .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
            .application_name = "test-texture-manager",
        };
        context = Context::create(info);
        auto resources = context->get_context_extension<ExtensionResources>();
        allocator = resources->resource_allocator();
        queue = context->get_queue_GCT();
        compile_context = ShaderCompileContext::create(context);
        compile_context->add_search_path(TEST_SHADER_DIR);
        obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);
    }

    static void TearDownTestSuite() {
        context->get_device()->get_device().waitIdle();
        obj_allocator.reset();
        compile_context.reset();
        allocator.reset();
        queue.reset();
        context.reset();
    }

    static uint32_t float_bits(float f) {
        return std::bit_cast<uint32_t>(f);
    }

    static float bits_float(uint32_t u) {
        return std::bit_cast<float>(u);
    }
};

ContextHandle TextureManagerTest::context;
ResourceAllocatorHandle TextureManagerTest::allocator;
QueueHandle TextureManagerTest::queue;
ShaderCompileContextHandle TextureManagerTest::compile_context;
ShaderObjectAllocatorHandle TextureManagerTest::obj_allocator;

TEST_F(TextureManagerTest, Construction) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator);
    EXPECT_EQ(tm->get_texture_count(), 0u);
    EXPECT_EQ(tm->get_capacity(), 4096u);
    EXPECT_NE(tm->get_composition(), nullptr);
}

TEST_F(TextureManagerTest, ConstructionWithCustomCapacity) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 128);
    EXPECT_EQ(tm->get_capacity(), 128u);
}

TEST_F(TextureManagerTest, AddAndRemoveTexture) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 16);

    auto tex = allocator->get_dummy_texture();
    TextureID id = tm->add_texture(tex);
    EXPECT_EQ(id, 0u);
    EXPECT_EQ(tm->get_texture_count(), 1u);
    EXPECT_EQ(tm->get_texture(id), tex);

    TextureID id2 = tm->add_texture(tex);
    EXPECT_EQ(id2, 1u);
    EXPECT_EQ(tm->get_texture_count(), 2u);

    tm->remove_texture(id);
    EXPECT_EQ(tm->get_texture_count(), 1u);

    // Reuse freed slot
    TextureID id3 = tm->add_texture(tex);
    EXPECT_EQ(id3, 0u);
    EXPECT_EQ(tm->get_texture_count(), 2u);
}

TEST_F(TextureManagerTest, SampleTextureOnGPU) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 16);

    // 1x1 red pixel (RGBA8, linear): R=255 G=0 B=0 A=255
    const uint32_t red_pixel = 0xFF0000FF;

    TextureID tex_id;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        tex_id = tm->add_texture_from_rgba8(cmd, &red_pixel, 1, 1, vk::SamplerAddressMode::eRepeat,
                                            vk::Filter::eNearest, vk::Filter::eNearest, false);
    });
    EXPECT_EQ(tex_id, 0u);

    // Build a composition: TextureManager modules + test compute shader
    auto composition = SlangComposition::create();
    composition->add_composition(tm->get_composition());
    composition->add_module_from_path("texture_manager/read_texture.slang", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);

    auto output_buffer = allocator->create_buffer(
        4 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto params = entry_point.get()->create_shader_object(context, "params", allocator);
    auto cursor = params->get_cursor();
    cursor["tm"] = tm;
    cursor["output"] = output_buffer;
    cursor["texture_id"] = static_cast<uint32_t>(tex_id);

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                      obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
    float r = bits_float(mapped[0]);
    float g = bits_float(mapped[1]);
    float b = bits_float(mapped[2]);
    float a = bits_float(mapped[3]);
    output_buffer->get_memory()->unmap();

    EXPECT_NEAR(r, 1.0f, 1e-3f);
    EXPECT_NEAR(g, 0.0f, 1e-3f);
    EXPECT_NEAR(b, 0.0f, 1e-3f);
    EXPECT_NEAR(a, 1.0f, 1e-3f);
}

TEST_F(TextureManagerTest, ResizePropagatesVersion) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 4);

    auto v0 = tm->version();
    auto comp_v0 = tm->get_composition()->version();

    // Fill to capacity
    auto dummy = allocator->get_dummy_texture();
    for (uint32_t i = 0; i < 4; i++) {
        tm->add_texture(dummy);
    }
    EXPECT_EQ(tm->version(), v0) << "No resize yet, version unchanged";

    // 5th texture triggers resize 4 → 8
    tm->add_texture(dummy);

    EXPECT_GT(tm->version(), v0) << "TextureManager version should increment on array resize";
    EXPECT_GT(tm->get_composition()->version(), comp_v0)
        << "Composition version should increment on array resize";
    EXPECT_EQ(tm->get_capacity(), 8u);
}
