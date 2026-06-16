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
#include <cstring>

using namespace merian;

#ifndef TEST_SHADER_DIR
#define TEST_SHADER_DIR "."
#endif

class MaterialSystemTest : public ::testing::Test {
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
            .application_name = "test-material-system",
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

ContextHandle MaterialSystemTest::context;
ResourceAllocatorHandle MaterialSystemTest::allocator;
QueueHandle MaterialSystemTest::queue;
ShaderCompileContextHandle MaterialSystemTest::compile_context;
ShaderObjectAllocatorHandle MaterialSystemTest::obj_allocator;

TEST_F(MaterialSystemTest, Construction) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator);
    auto ms = std::make_shared<MaterialSystem>(compile_context, context, allocator, tm);
    EXPECT_EQ(ms->get_material_count(), 0u);
    EXPECT_NE(ms->get_composition(), nullptr);
    EXPECT_EQ(ms->get_texture_manager(), tm);
}

TEST_F(MaterialSystemTest, RegisterAndAddMaterials) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator);
    auto ms = std::make_shared<MaterialSystem>(compile_context, context, allocator, tm);

    auto type_id = ms->register_material_type(
        "merian::DiffuseMaterial", "merian-shaders/shading/materials/diffuse-material.slang");
    EXPECT_EQ(type_id, 0u);

    DiffuseMaterial mat1(float4(1, 0, 0, 1), TextureID(-1));
    MaterialID id1 = ms->add_material(type_id, mat1);
    EXPECT_EQ(id1, 0u);
    EXPECT_EQ(ms->get_material_count(), 1u);

    DiffuseMaterial mat2(float4(0, 1, 0, 1), TextureID(-1));
    MaterialID id2 = ms->add_material(type_id, mat2);
    EXPECT_EQ(id2, 1u);
    EXPECT_EQ(ms->get_material_count(), 2u);
}

TEST_F(MaterialSystemTest, ReadMaterialOnGPU) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 16);
    auto ms = std::make_shared<MaterialSystem>(compile_context, context, allocator, tm);

    auto type_id = ms->register_material_type(
        "merian::DiffuseMaterial", "merian-shaders/shading/materials/diffuse-material.slang");

    DiffuseMaterial mat(float4(0.25f, 0.5f, 0.75f, 1.0f), TextureID(-1));
    MaterialID mat_id = ms->add_material(type_id, mat);

    auto composition = SlangComposition::create();
    composition->add_composition(ms->get_composition());
    composition->add_module_from_path("material_system/read_material.slang", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);

    auto output_buffer = allocator->create_buffer(
        7 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto params =
        entry_point.get()->create_shader_object_for_parameter(context, "params", allocator);
    auto cursor = params->get_cursor();
    cursor["ms"] = ms;
    cursor["output"] = output_buffer;
    cursor["material_id"] = static_cast<uint32_t>(mat_id);

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        ms->update(cmd);
        cmd->bind(pipeline);
        entry_point.get()->bind("params", params, cmd, pipeline, obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
    uint32_t type_id_read = mapped[0];
    uint32_t alpha_tex_read = mapped[1];
    float payload_r = bits_float(mapped[2]);
    float payload_g = bits_float(mapped[3]);
    float payload_b = bits_float(mapped[4]);
    float payload_a = bits_float(mapped[5]);
    uint32_t material_count = mapped[6];
    output_buffer->get_memory()->unmap();

    EXPECT_EQ(type_id_read, static_cast<uint32_t>(type_id));
    EXPECT_EQ(alpha_tex_read, static_cast<uint32_t>(TextureID(-1)));
    EXPECT_NEAR(payload_r, 0.25f, 1e-5f);
    EXPECT_NEAR(payload_g, 0.5f, 1e-5f);
    EXPECT_NEAR(payload_b, 0.75f, 1e-5f);
    EXPECT_NEAR(payload_a, 1.0f, 1e-5f);
    EXPECT_EQ(material_count, 1u);
}

TEST_F(MaterialSystemTest, SampleMaterialOnGPU) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 16);
    auto ms = std::make_shared<MaterialSystem>(compile_context, context, allocator, tm);

    auto type_id = ms->register_material_type(
        "merian::DiffuseMaterial", "merian-shaders/shading/materials/diffuse-material.slang");

    DiffuseMaterial mat(float4(1.0f, 0.0f, 0.0f, 1.0f), TextureID(-1));
    MaterialID mat_id = ms->add_material(type_id, mat);

    auto composition = SlangComposition::create();
    composition->add_composition(ms->get_composition());
    composition->add_module_from_path("material_system/sample_material.slang", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);

    auto output_buffer = allocator->create_buffer(
        3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto params =
        entry_point.get()->create_shader_object_for_parameter(context, "params", allocator);
    auto cursor = params->get_cursor();
    cursor["ms"] = ms;
    cursor["output"] = output_buffer;
    cursor["material_id"] = static_cast<uint32_t>(mat_id);

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        ms->update(cmd);
        cmd->bind(pipeline);
        entry_point.get()->bind("params", params, cmd, pipeline, obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
    float r = bits_float(mapped[0]);
    float g = bits_float(mapped[1]);
    float b = bits_float(mapped[2]);
    output_buffer->get_memory()->unmap();

    // Lambert BRDF eval(wi=(0,0,1), wo=(0,0,1)) = INV_PI * albedo * wo.z = INV_PI * albedo
    const float inv_pi = 1.0f / 3.14159265358979323846f;
    EXPECT_NEAR(r, inv_pi * 1.0f, 1e-5f);
    EXPECT_NEAR(g, inv_pi * 0.0f, 1e-5f);
    EXPECT_NEAR(b, inv_pi * 0.0f, 1e-5f);
}

TEST_F(MaterialSystemTest, PayloadResizePropagatesVersion) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 4);
    auto ms = std::make_shared<MaterialSystem>(compile_context, context, allocator, tm);

    auto type_id = ms->register_material_type(
        "merian::DiffuseMaterial", "merian-shaders/shading/materials/diffuse-material.slang");

    auto v0 = ms->version();
    auto comp_v0 = ms->get_composition()->version();

    // Adding first material triggers payload resize (0 → DiffuseMaterial payload size)
    DiffuseMaterial mat(float4(1, 0, 0, 1), TextureID(-1));
    ms->add_material(type_id, mat);

    EXPECT_GT(ms->version(), v0) << "MaterialSystem version should increment on payload resize";
    EXPECT_GT(ms->get_composition()->version(), comp_v0)
        << "Composition version should increment on payload resize";
}

TEST_F(MaterialSystemTest, TextureManagerResizePropagatesToMaterialSystem) {
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 4);
    auto ms = std::make_shared<MaterialSystem>(compile_context, context, allocator, tm);

    auto ms_comp_v0 = ms->get_composition()->version();

    // Fill TM to capacity, then trigger resize
    auto dummy = allocator->get_dummy_texture();
    for (uint32_t i = 0; i < 4; i++) {
        tm->add_texture(dummy);
    }
    tm->add_texture(dummy); // triggers resize 4 → 8

    EXPECT_GT(ms->get_composition()->version(), ms_comp_v0)
        << "MaterialSystem composition should see TM sub-composition change";
}
