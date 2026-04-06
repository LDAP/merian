#include <gtest/gtest.h>

#include "merian-shaders/shading/materials/material_system.hpp"
#include "merian-shaders/utils/texture_manager.hpp"
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

using namespace merian;

#ifndef TEST_SHADER_DIR
#define TEST_SHADER_DIR "."
#endif

class SlangHotReloadTest : public ::testing::Test {
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
            .application_name = "test-slang-hot-reload",
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

    static float bits_float(uint32_t u) {
        return std::bit_cast<float>(u);
    }
};

ContextHandle SlangHotReloadTest::context;
ResourceAllocatorHandle SlangHotReloadTest::allocator;
QueueHandle SlangHotReloadTest::queue;
ShaderCompileContextHandle SlangHotReloadTest::compile_context;
ShaderObjectAllocatorHandle SlangHotReloadTest::obj_allocator;

// ---------------------------------------------------------------------------
// End-to-end: material system payload resize → full pipeline rebuild on GPU
// ---------------------------------------------------------------------------

TEST_F(SlangHotReloadTest, MaterialPipelineRebuildsAfterForceReload) {
    auto tm =
        std::make_shared<TextureManager>(compile_context, context, allocator, obj_allocator, 16);
    auto ms =
        std::make_shared<MaterialSystem>(compile_context, context, allocator, obj_allocator, tm);

    auto type_id = ms->register_material_type(
        "merian::DiffuseMaterial",
        "merian-shaders/shading/materials/diffuse-material.slang");

    DiffuseMaterial mat1(float4(0.25f, 0.5f, 0.75f, 1.0f), TextureID(-1));
    MaterialID mat1_id = ms->add_material(type_id, mat1);

    // Build the pipeline
    auto composition = SlangComposition::create();
    composition->add_composition(ms->get_composition());
    composition->add_module_from_path("material_system/read_material.slang", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");

    // Auto-rebuild pipeline via listener on entry point
    PipelineLayoutHandle pipe_layout;
    PipelineHandle pipeline;
    uint32_t rebuild_count = 0;

    auto build_pipeline = [&]() {
        pipe_layout = entry_point->get_pipeline_layout(context);
        pipeline = ComputePipeline::create(pipe_layout, entry_point->specialize());
        rebuild_count++;
    };
    build_pipeline(); // initial build

    entry_point->on_changed(entry_point, [&]() { build_pipeline(); });

    // Run first dispatch
    {
        auto output_buffer = allocator->create_buffer(
            7 * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::HOST_ACCESS_RANDOM, "test_output_1");

        auto params = entry_point->create_shader_object(context, "params", obj_allocator);
        params->get_cursor()["ms"] = ms;
        params->get_cursor()["output"] = output_buffer;
        params->get_cursor()["material_id"] = static_cast<uint32_t>(mat1_id);

        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            ms->upload(cmd);
            cmd->bind(pipeline);
            entry_point->bind_entry_point_parameter("params", params, cmd, pipeline);
            cmd->dispatch(1, 1, 1);
        });

        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        EXPECT_EQ(mapped[0], static_cast<uint32_t>(type_id));
        EXPECT_NEAR(bits_float(mapped[2]), 0.25f, 1e-5f);
        EXPECT_NEAR(bits_float(mapped[3]), 0.5f, 1e-5f);
        EXPECT_NEAR(bits_float(mapped[4]), 0.75f, 1e-5f);
        EXPECT_NEAR(bits_float(mapped[5]), 1.0f, 1e-5f);
        EXPECT_EQ(mapped[6], 1u);
        output_buffer->get_memory()->unmap();
    }

    // Force reload — listener should automatically rebuild pipeline
    EXPECT_EQ(rebuild_count, 1u);
    composition->force_reload();
    EXPECT_EQ(rebuild_count, 2u) << "Listener should have rebuilt the pipeline";

    // Add a second material and dispatch with the auto-rebuilt pipeline
    DiffuseMaterial mat2(float4(0.1f, 0.2f, 0.3f, 0.4f), TextureID(-1));
    MaterialID mat2_id = ms->add_material(type_id, mat2);

    {
        auto output_buffer = allocator->create_buffer(
            7 * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::HOST_ACCESS_RANDOM, "test_output_2");

        auto params = entry_point->create_shader_object(context, "params", obj_allocator);
        params->get_cursor()["ms"] = ms;
        params->get_cursor()["output"] = output_buffer;
        params->get_cursor()["material_id"] = static_cast<uint32_t>(mat2_id);

        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            ms->upload(cmd);
            cmd->bind(pipeline);
            entry_point->bind_entry_point_parameter("params", params, cmd, pipeline);
            cmd->dispatch(1, 1, 1);
        });

        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        EXPECT_EQ(mapped[0], static_cast<uint32_t>(type_id));
        EXPECT_NEAR(bits_float(mapped[2]), 0.1f, 1e-5f);
        EXPECT_NEAR(bits_float(mapped[3]), 0.2f, 1e-5f);
        EXPECT_NEAR(bits_float(mapped[4]), 0.3f, 1e-5f);
        EXPECT_NEAR(bits_float(mapped[5]), 0.4f, 1e-5f);
        EXPECT_EQ(mapped[6], 2u);
        output_buffer->get_memory()->unmap();
    }
}

// ---------------------------------------------------------------------------
// End-to-end: texture manager array resize → pipeline rebuild → GPU read
// ---------------------------------------------------------------------------

TEST_F(SlangHotReloadTest, TextureArrayResizeRebuildsFullPipeline) {
    // Start with capacity 4
    auto tm =
        std::make_shared<TextureManager>(compile_context, context, allocator, obj_allocator, 4);

    // Build pipeline using TM composition
    auto composition = SlangComposition::create();
    composition->add_composition(tm->get_composition());
    composition->add_module_from_path("texture_manager/read_texture.slang", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");

    // Auto-rebuild pipeline via listener on entry point
    PipelineLayoutHandle pipe_layout;
    PipelineHandle pipeline;
    uint32_t rebuild_count = 0;

    auto build_pipeline = [&]() {
        pipe_layout = entry_point->get_pipeline_layout(context);
        pipeline = ComputePipeline::create(pipe_layout, entry_point->specialize());
        rebuild_count++;
    };
    build_pipeline(); // initial build

    entry_point->on_changed(entry_point, [&]() { build_pipeline(); });

    // Fill to capacity
    auto dummy = allocator->get_dummy_texture();
    for (uint32_t i = 0; i < 4; i++) {
        tm->add_texture(dummy);
    }
    EXPECT_EQ(rebuild_count, 1u) << "No resize yet, pipeline unchanged";

    // Add a real texture — triggers resize 4 → 8 → listener rebuilds pipeline
    const uint32_t red_pixel = 0xFF0000FF;
    TextureID tex_id;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        tex_id = tm->add_texture_from_rgba8(cmd, &red_pixel, 1, 1, vk::Filter::eNearest, false);
    });
    EXPECT_EQ(tm->get_capacity(), 8u);
    EXPECT_EQ(rebuild_count, 2u) << "Listener should have rebuilt pipeline after TM resize";

    // Dispatch with the auto-rebuilt pipeline
    auto output_buffer = allocator->create_buffer(
        4 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto params = entry_point->create_shader_object(context, "params", obj_allocator);
    params->get_cursor()["tm"] = tm;
    params->get_cursor()["output"] = output_buffer;
    params->get_cursor()["texture_id"] = static_cast<uint32_t>(tex_id);

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point->bind_entry_point_parameter("params", params, cmd, pipeline);
        cmd->dispatch(1, 1, 1);
    });

    auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
    EXPECT_NEAR(bits_float(mapped[0]), 1.0f, 1e-3f); // R
    EXPECT_NEAR(bits_float(mapped[1]), 0.0f, 1e-3f); // G
    EXPECT_NEAR(bits_float(mapped[2]), 0.0f, 1e-3f); // B
    EXPECT_NEAR(bits_float(mapped[3]), 1.0f, 1e-3f); // A
    output_buffer->get_memory()->unmap();
}
