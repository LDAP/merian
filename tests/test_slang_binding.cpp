#include <gtest/gtest.h>

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

#include <bit>
#include <cstring>
#include <filesystem>

using namespace merian;

// Path to shader test files, set from meson build directory
#ifndef SLANG_BINDING_TEST_SHADER_DIR
#define SLANG_BINDING_TEST_SHADER_DIR "."
#endif

// ---------------------------------------------------------------------------
// Test fixture — shared Vulkan context across all tests
// ---------------------------------------------------------------------------

class SlangBindingTest : public ::testing::Test {
  protected:
    static ContextHandle context;
    static ResourceAllocatorHandle allocator;
    static QueueHandle queue;
    static ShaderCompileContextHandle compile_context;

    static void SetUpTestSuite() {
        spdlog::set_level(spdlog::level::debug);
        ContextCreateInfo info{
            .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
            .application_name = "test-slang-binding",
        };
        context = Context::create(info);
        auto resources = context->get_context_extension<ExtensionResources>();
        allocator = resources->resource_allocator();
        queue = context->get_queue_GCT();
        compile_context = ShaderCompileContext::create(context);
        compile_context->add_search_path(SLANG_BINDING_TEST_SHADER_DIR);
    }

    static void TearDownTestSuite() {
        context->get_device()->get_device().waitIdle();
        compile_context.reset();
        allocator.reset();
        queue.reset();
        context.reset();
    }

    // -----------------------------------------------------------------------
    // Helper: compile shader, set params via single PB, dispatch, read back
    // -----------------------------------------------------------------------

    struct TestResult {
        std::vector<uint32_t> data;
    };

    static TestResult
    dispatch_and_readback(const std::string& shader_name,
                          uint32_t output_slots,
                          const std::function<void(const SlangProgramEntryPointHandle& entry_point,
                                                   ShaderCursor& cursor,
                                                   const BufferHandle& output_buffer)>& setup_fn) {

        // Compile
        auto program =
            SlangProgram::create(compile_context, "slang_binding/" + shader_name + ".slang");
        auto entry_point = SlangProgramEntryPoint::create(program, "main");
        auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
        auto vulkan_ep = entry_point.get()->specialize();
        auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);

        // Allocator
        auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

        // Output buffer — host-visible for direct readback
        const vk::DeviceSize output_size = output_slots * sizeof(uint32_t);
        auto output_buffer = allocator->create_buffer(
            output_size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

        // Create params and set values
        auto params = entry_point.get()->create_shader_object(context, "params", allocator);
        SPDLOG_DEBUG("ShaderObject:\n{}", *params);
        auto cursor = params->get_cursor();
        setup_fn(entry_point.get(), cursor, output_buffer);

        // Dispatch
        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            cmd->bind(pipeline);
            entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                          obj_allocator);
            cmd->dispatch(1, 1, 1);
        });

        // Readback
        TestResult result;
        result.data.resize(output_slots);
        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        std::memcpy(result.data.data(), mapped, output_size);
        output_buffer->get_memory()->unmap();

        return result;
    }

    // Helper for multi-PB tests: provides full pipeline + entry point access
    struct MultiPBContext {
        SlangProgramEntryPointHandle entry_point;
        PipelineHandle pipeline;
        ShaderObjectAllocatorHandle obj_allocator;
        BufferHandle output_buffer;
    };

    static MultiPBContext create_multi_pb_context(const std::string& shader_name,
                                                  uint32_t output_slots) {
        auto program =
            SlangProgram::create(compile_context, "slang_binding/" + shader_name + ".slang");
        const SlangProgramEntryPointHandle entry_point =
            SlangProgramEntryPoint::create(program, "main").get();
        auto pipe_layout = entry_point->get_pipeline_layout(context);
        auto vulkan_ep = entry_point->specialize();
        auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);
        auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

        const vk::DeviceSize output_size = output_slots * sizeof(uint32_t);
        auto output_buffer = allocator->create_buffer(
            output_size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

        return {entry_point, pipeline, obj_allocator, output_buffer};
    }

    static TestResult readback(const BufferHandle& buffer, uint32_t slots) {
        TestResult result;
        result.data.resize(slots);
        auto* mapped = buffer->get_memory()->map_as<uint32_t>();
        std::memcpy(result.data.data(), mapped, slots * sizeof(uint32_t));
        buffer->get_memory()->unmap();
        return result;
    }

    static uint32_t float_bits(float f) {
        return std::bit_cast<uint32_t>(f);
    }

    static uint32_t int_bits(int32_t i) {
        return std::bit_cast<uint32_t>(i);
    }
};

// Static member definitions
ContextHandle SlangBindingTest::context;
ResourceAllocatorHandle SlangBindingTest::allocator;
QueueHandle SlangBindingTest::queue;
ShaderCompileContextHandle SlangBindingTest::compile_context;

// ===========================================================================
// 1. Value bindings
// ===========================================================================

TEST_F(SlangBindingTest, ScalarTypes) {
    auto result = dispatch_and_readback(
        "scalar_types", 3,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            cursor["val_float"] = 3.14f;
            cursor["val_int"] = -42;
            cursor["val_uint"] = 123u;
            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(3.14f));
    EXPECT_EQ(result.data[1], int_bits(-42));
    EXPECT_EQ(result.data[2], 123u);
}

TEST_F(SlangBindingTest, VectorTypes) {
    struct float2 {
        float x, y;
    };
    struct float3 {
        float x, y, z;
    };
    struct float4 {
        float x, y, z, w;
    };
    struct int3 {
        int32_t x, y, z;
    };

    auto result = dispatch_and_readback(
        "vector_types", 12,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            cursor["val_float2"] = float2{1.0f, 2.0f};
            cursor["val_float3"] = float3{3.0f, 4.0f, 5.0f};
            cursor["val_float4"] = float4{6.0f, 7.0f, 8.0f, 9.0f};
            cursor["val_int3"] = int3{-1, -2, -3};
            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(1.0f));
    EXPECT_EQ(result.data[1], float_bits(2.0f));
    EXPECT_EQ(result.data[2], float_bits(3.0f));
    EXPECT_EQ(result.data[3], float_bits(4.0f));
    EXPECT_EQ(result.data[4], float_bits(5.0f));
    EXPECT_EQ(result.data[5], float_bits(6.0f));
    EXPECT_EQ(result.data[6], float_bits(7.0f));
    EXPECT_EQ(result.data[7], float_bits(8.0f));
    EXPECT_EQ(result.data[8], float_bits(9.0f));
    EXPECT_EQ(result.data[9], int_bits(-1));
    EXPECT_EQ(result.data[10], int_bits(-2));
    EXPECT_EQ(result.data[11], int_bits(-3));
}

TEST_F(SlangBindingTest, MatrixType) {
    // float4x4 — 16 floats, row-major
    float mat[16];
    for (int i = 0; i < 16; i++)
        mat[i] = static_cast<float>(i + 1);

    auto result = dispatch_and_readback("matrix_type", 16,
                                        [&mat](const SlangProgramEntryPointHandle&,
                                               ShaderCursor& cursor, const BufferHandle& output) {
                                            cursor["val_mat"].write(mat, sizeof(mat));
                                            cursor["output"] = output;
                                        });

    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(result.data[i], float_bits(static_cast<float>(i + 1))) << "index " << i;
    }
}

TEST_F(SlangBindingTest, EmbeddedStruct) {
    struct Inner {
        float a;
        int32_t b;
    };

    auto result = dispatch_and_readback(
        "embedded_struct", 3,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            cursor["inner"]["a"] = 2.5f;
            cursor["inner"]["b"] = -7;
            cursor["c"] = 42u;
            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(2.5f));
    EXPECT_EQ(result.data[1], int_bits(-7));
    EXPECT_EQ(result.data[2], 42u);
}

// ===========================================================================
// 2. ConstantBuffer
// ===========================================================================

TEST_F(SlangBindingTest, SingleCB) {
    auto result = dispatch_and_readback(
        "single_cb", 5,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            // cursor["cb"] auto-creates the CB sub-object and returns cursor into element type
            cursor["cb"]["x"] = 1.5f;
            cursor["cb"]["y"] = -3;
            struct float3 {
                float x, y, z;
            };
            cursor["cb"]["z"] = float3{4.0f, 5.0f, 6.0f};
            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(1.5f));
    EXPECT_EQ(result.data[1], int_bits(-3));
    EXPECT_EQ(result.data[2], float_bits(4.0f));
    EXPECT_EQ(result.data[3], float_bits(5.0f));
    EXPECT_EQ(result.data[4], float_bits(6.0f));
}

TEST_F(SlangBindingTest, MultipleCBs) {
    auto result = dispatch_and_readback(
        "multiple_cbs", 4,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            cursor["cb_a"]["a"] = 1.0f;
            cursor["cb_a"]["b"] = 99u;
            cursor["cb_b"]["c"] = -5;
            cursor["cb_b"]["d"] = 2.0f;
            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(1.0f));
    EXPECT_EQ(result.data[1], 99u);
    EXPECT_EQ(result.data[2], int_bits(-5));
    EXPECT_EQ(result.data[3], float_bits(2.0f));
}

TEST_F(SlangBindingTest, CBInCB) {
    auto result = dispatch_and_readback(
        "cb_in_cb", 3,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            cursor["outer"]["inner"]["a"] = 7.0f;
            cursor["outer"]["inner"]["b"] = -13;
            cursor["outer"]["c"] = 55u;
            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(7.0f));
    EXPECT_EQ(result.data[1], int_bits(-13));
    EXPECT_EQ(result.data[2], 55u);
}

TEST_F(SlangBindingTest, CBInCBDepth3) {
    auto result = dispatch_and_readback(
        "cb_in_cb_depth3", 3,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            cursor["level1"]["level2"]["level3"]["a"] = 3.14f;
            cursor["level1"]["level2"]["b"] = -100;
            cursor["level1"]["c"] = 200u;
            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(3.14f));
    EXPECT_EQ(result.data[1], int_bits(-100));
    EXPECT_EQ(result.data[2], 200u);
}

// ===========================================================================
// 3. Resource types
// ===========================================================================

TEST_F(SlangBindingTest, StorageBuffer) {
    // Create an input buffer with known data
    auto input_buffer = allocator->create_buffer(
        3 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_input");
    {
        auto* mapped = input_buffer->get_memory()->map_as<uint32_t>();
        mapped[0] = 10;
        mapped[1] = 20;
        mapped[2] = 30;
        input_buffer->get_memory()->unmap();
    }

    auto result =
        dispatch_and_readback("storage_buffer", 3,
                              [&input_buffer](const SlangProgramEntryPointHandle&,
                                              ShaderCursor& cursor, const BufferHandle& output) {
                                  cursor["input"] = input_buffer;
                                  cursor["output"] = output;
                              });

    EXPECT_EQ(result.data[0], 11u); // 10 + 1
    EXPECT_EQ(result.data[1], 40u); // 20 * 2
    EXPECT_EQ(result.data[2], 30u); // 30
}

// ===========================================================================
// 4. Multiple entry point ParameterBlocks
// ===========================================================================

TEST_F(SlangBindingTest, TwoEntryPointPBs) {
    auto ctx = create_multi_pb_context("two_entry_point_pbs", 4);

    auto pa = ctx.entry_point->create_shader_object(context, "pa", allocator);
    auto cursor_a = pa->get_cursor();
    cursor_a["a"] = 1.5f;
    cursor_a["b"] = 77u;
    cursor_a["output"] = ctx.output_buffer;

    auto pb = ctx.entry_point->create_shader_object(context, "pb", allocator);
    auto cursor_b = pb->get_cursor();
    cursor_b["c"] = -8;
    cursor_b["d"] = 2.5f;

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(ctx.pipeline);
        ctx.entry_point->bind_entry_point_parameter("pa", pa, cmd, ctx.pipeline, ctx.obj_allocator);
        ctx.entry_point->bind_entry_point_parameter("pb", pb, cmd, ctx.pipeline, ctx.obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto result = readback(ctx.output_buffer, 4);
    EXPECT_EQ(result.data[0], float_bits(1.5f));
    EXPECT_EQ(result.data[1], 77u);
    EXPECT_EQ(result.data[2], int_bits(-8));
    EXPECT_EQ(result.data[3], float_bits(2.5f));
}

// ===========================================================================
// 5. Incremental updates
// ===========================================================================

TEST_F(SlangBindingTest, IncrementalValueUpdate) {
    auto program = SlangProgram::create(compile_context, "slang_binding/incremental_value.slang");
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);
    auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

    auto output_buffer =
        allocator->create_buffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
                                 MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto params = entry_point.get()->create_shader_object(context, "params", allocator);
    auto cursor = params->get_cursor();
    cursor["output"] = output_buffer;

    // First dispatch with val = 1.0
    cursor["val"] = 1.0f;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                      obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    {
        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        EXPECT_EQ(mapped[0], float_bits(1.0f)) << "First dispatch";
        output_buffer->get_memory()->unmap();
    }

    // Second dispatch with val = 2.0 — same ShaderObject, just update the value
    cursor["val"] = 2.0f;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                      obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    {
        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        EXPECT_EQ(mapped[0], float_bits(2.0f)) << "Second dispatch with updated value";
        output_buffer->get_memory()->unmap();
    }
}

// ===========================================================================
// 6. Manual sub-object API
// ===========================================================================

TEST_F(SlangBindingTest, ManualCBAssignment) {
    auto result = dispatch_and_readback(
        "single_cb", 5,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            // Use create_subobject + set_subobject instead of cursor.dereference()
            auto base_obj = cursor.get_base_object();
            auto cb_obj = base_obj->create_subobject("cb");
            auto cb_cursor = cb_obj->get_cursor();
            cb_cursor["x"] = 10.0f;
            cb_cursor["y"] = 20;
            struct float3 {
                float x, y, z;
            };
            cb_cursor["z"] = float3{30.0f, 40.0f, 50.0f};
            base_obj->set_subobject("cb", cb_obj);
            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(10.0f));
    EXPECT_EQ(result.data[1], int_bits(20));
    EXPECT_EQ(result.data[2], float_bits(30.0f));
    EXPECT_EQ(result.data[3], float_bits(40.0f));
    EXPECT_EQ(result.data[4], float_bits(50.0f));
}

TEST_F(SlangBindingTest, SubObjectReassignment) {
    // Create first CB, then replace it with a second one — verify second values used
    auto result = dispatch_and_readback(
        "single_cb", 5,
        [](const SlangProgramEntryPointHandle&, ShaderCursor& cursor, const BufferHandle& output) {
            auto base_obj = cursor.get_base_object();

            // First assignment
            auto cb1 = base_obj->create_subobject("cb");
            {
                auto c = cb1->get_cursor();
                c["x"] = 1.0f;
                c["y"] = 1;
                struct float3 {
                    float x, y, z;
                };
                c["z"] = float3{1.0f, 1.0f, 1.0f};
            }
            base_obj->set_subobject("cb", cb1);

            // Reassign with different values
            auto cb2 = base_obj->create_subobject("cb");
            {
                auto c = cb2->get_cursor();
                c["x"] = 99.0f;
                c["y"] = -99;
                struct float3 {
                    float x, y, z;
                };
                c["z"] = float3{88.0f, 77.0f, 66.0f};
            }
            base_obj->set_subobject("cb", cb2);

            cursor["output"] = output;
        });

    EXPECT_EQ(result.data[0], float_bits(99.0f));
    EXPECT_EQ(result.data[1], int_bits(-99));
    EXPECT_EQ(result.data[2], float_bits(88.0f));
    EXPECT_EQ(result.data[3], float_bits(77.0f));
    EXPECT_EQ(result.data[4], float_bits(66.0f));
}

// ===========================================================================
// 7. ParameterBlock nesting
// ===========================================================================

TEST_F(SlangBindingTest, SingleNestedPB) {
    auto ctx = create_multi_pb_context("single_nested_pb", 3);

    auto params = ctx.entry_point->create_shader_object(context, "params", allocator);
    auto cursor = params->get_cursor();
    cursor["c"] = 42u;

    // The nested PB is a separate ShaderObject bound at its own descriptor set
    auto inner = ctx.entry_point->create_shader_object(context, "params", allocator);
    // Actually, the nested PB ("inner") is a sub-object of params, not a separate entry point PB.
    // We need to create it through the params object.
    auto inner_obj = params->create_subobject("inner");
    auto inner_cursor = inner_obj->get_cursor();
    inner_cursor["a"] = 5.0f;
    inner_cursor["b"] = -10;
    inner_cursor["output"] = ctx.output_buffer;
    params->set_subobject("inner", inner_obj);

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(ctx.pipeline);
        ctx.entry_point->bind_entry_point_parameter("params", params, cmd, ctx.pipeline,
                                                    ctx.obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto result = readback(ctx.output_buffer, 3);
    EXPECT_EQ(result.data[0], float_bits(5.0f));
    EXPECT_EQ(result.data[1], int_bits(-10));
    EXPECT_EQ(result.data[2], 42u);
}

TEST_F(SlangBindingTest, PBWithCB) {
    auto ctx = create_multi_pb_context("pb_with_cb", 3);

    auto params = ctx.entry_point->create_shader_object(context, "params", allocator);

    // The nested PB "inner" needs manual setup since it contains a CB
    auto inner_obj = params->create_subobject("inner");
    auto inner_cursor = inner_obj->get_cursor();
    inner_cursor["cb"]["x"] = 3.0f;
    inner_cursor["cb"]["y"] = -7;
    inner_cursor["z"] = 99u;
    inner_cursor["output"] = ctx.output_buffer;
    params->set_subobject("inner", inner_obj);

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(ctx.pipeline);
        ctx.entry_point->bind_entry_point_parameter("params", params, cmd, ctx.pipeline,
                                                    ctx.obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto result = readback(ctx.output_buffer, 3);
    EXPECT_EQ(result.data[0], float_bits(3.0f));
    EXPECT_EQ(result.data[1], int_bits(-7));
    EXPECT_EQ(result.data[2], 99u);
}

// ===========================================================================
// 8. Incremental updates (CB and resource)
// ===========================================================================

TEST_F(SlangBindingTest, IncrementalCBUpdate) {
    auto program = SlangProgram::create(compile_context, "slang_binding/incremental_cb.slang");
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);
    auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

    auto output_buffer =
        allocator->create_buffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
                                 MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto params = entry_point.get()->create_shader_object(context, "params", allocator);
    auto cursor = params->get_cursor();
    cursor["output"] = output_buffer;

    // First dispatch
    cursor["cb"]["val"] = 1.0f;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                      obj_allocator);
        cmd->dispatch(1, 1, 1);
    });
    {
        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        EXPECT_EQ(mapped[0], float_bits(1.0f)) << "First dispatch";
        output_buffer->get_memory()->unmap();
    }

    // Second dispatch with updated CB value
    cursor["cb"]["val"] = 2.0f;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                      obj_allocator);
        cmd->dispatch(1, 1, 1);
    });
    {
        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        EXPECT_EQ(mapped[0], float_bits(2.0f)) << "Second dispatch with updated CB value";
        output_buffer->get_memory()->unmap();
    }
}

TEST_F(SlangBindingTest, IncrementalResourceUpdate) {
    auto program =
        SlangProgram::create(compile_context, "slang_binding/incremental_resource.slang");
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);
    auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

    auto output_buffer =
        allocator->create_buffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
                                 MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto input_a =
        allocator->create_buffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
                                 MemoryMappingType::HOST_ACCESS_RANDOM, "input_a");
    auto input_b =
        allocator->create_buffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
                                 MemoryMappingType::HOST_ACCESS_RANDOM, "input_b");

    {
        auto* m = input_a->get_memory()->map_as<uint32_t>();
        m[0] = 100;
        input_a->get_memory()->unmap();
    }
    {
        auto* m = input_b->get_memory()->map_as<uint32_t>();
        m[0] = 200;
        input_b->get_memory()->unmap();
    }

    auto params = entry_point.get()->create_shader_object(context, "params", allocator);
    auto cursor = params->get_cursor();
    cursor["output"] = output_buffer;

    // First dispatch with input_a
    cursor["input"] = input_a;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                      obj_allocator);
        cmd->dispatch(1, 1, 1);
    });
    {
        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        EXPECT_EQ(mapped[0], 100u) << "First dispatch with input_a";
        output_buffer->get_memory()->unmap();
    }

    // Second dispatch with input_b — swap buffer
    cursor["input"] = input_b;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                      obj_allocator);
        cmd->dispatch(1, 1, 1);
    });
    {
        auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
        EXPECT_EQ(mapped[0], 200u) << "Second dispatch with input_b";
        output_buffer->get_memory()->unmap();
    }
}

// ===========================================================================
// 9. Combinations
// ===========================================================================

TEST_F(SlangBindingTest, FullMix) {
    auto input_buffer =
        allocator->create_buffer(2 * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
                                 MemoryMappingType::HOST_ACCESS_RANDOM, "test_input");
    {
        auto* m = input_buffer->get_memory()->map_as<uint32_t>();
        m[0] = 10;
        m[1] = 20;
        input_buffer->get_memory()->unmap();
    }

    auto result =
        dispatch_and_readback("full_mix", 5,
                              [&input_buffer](const SlangProgramEntryPointHandle&,
                                              ShaderCursor& cursor, const BufferHandle& output) {
                                  cursor["val"] = 1.5f;
                                  cursor["count"] = 42u;
                                  cursor["cb"]["a"] = 3.0f;
                                  cursor["cb"]["b"] = -7;
                                  cursor["input"] = input_buffer;
                                  cursor["output"] = output;
                              });

    EXPECT_EQ(result.data[0], float_bits(1.5f));
    EXPECT_EQ(result.data[1], 42u);
    EXPECT_EQ(result.data[2], float_bits(3.0f));
    EXPECT_EQ(result.data[3], int_bits(-7));
    EXPECT_EQ(result.data[4], 30u); // 10 + 20
}

// ===========================================================================
// 10. Global parameters
// ===========================================================================

TEST_F(SlangBindingTest, GlobalResource) {
    auto program = SlangProgram::create(compile_context, "slang_binding/global_resource.slang");
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);
    auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

    auto output_buffer = allocator->create_buffer(
        2 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    ASSERT_TRUE(entry_point.get()->has_globals(context));
    ASSERT_EQ(entry_point.get()->get_global_set_count(), 1u);

    auto globals = entry_point.get()->create_global_shader_object(context, allocator);
    auto cursor = globals->get_cursor();
    cursor["cb"]["x"] = 3.14f;
    cursor["cb"]["y"] = -42;
    cursor["output"] = output_buffer;

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_global_parameter(globals, cmd, pipeline, obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto result = readback(output_buffer, 2);
    EXPECT_EQ(result.data[0], float_bits(3.14f));
    EXPECT_EQ(result.data[1], int_bits(-42));
}

TEST_F(SlangBindingTest, GlobalPB) {
    auto program = SlangProgram::create(compile_context, "slang_binding/global_pb.slang");
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);
    auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

    auto output_buffer = allocator->create_buffer(
        2 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto globals = entry_point.get()->create_global_shader_object(context, allocator);
    auto cursor = globals->get_cursor();
    cursor["params"]["x"] = 1.5f;
    cursor["params"]["y"] = -7;
    cursor["params"]["output"] = output_buffer;

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_global_parameter(globals, cmd, pipeline, obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto result = readback(output_buffer, 2);
    EXPECT_EQ(result.data[0], float_bits(1.5f));
    EXPECT_EQ(result.data[1], int_bits(-7));
}

TEST_F(SlangBindingTest, GlobalWithPB) {
    auto program = SlangProgram::create(compile_context, "slang_binding/global_with_pb.slang");
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    auto pipe_layout = entry_point.get()->get_pipeline_layout(context);
    auto vulkan_ep = entry_point.get()->specialize();
    auto pipeline = ComputePipeline::create(pipe_layout, vulkan_ep);
    auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

    auto output_buffer = allocator->create_buffer(
        2 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    ASSERT_TRUE(entry_point.get()->has_globals(context));

    // Set global resources
    auto globals = entry_point.get()->create_global_shader_object(context, allocator);
    globals->get_cursor()["g_output"] = output_buffer;

    // Set PB params
    auto params = entry_point.get()->create_shader_object(context, "params", allocator);
    auto cursor = params->get_cursor();
    cursor["x"] = 1.5f;
    cursor["y"] = -7;

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline);
        entry_point.get()->bind_global_parameter(globals, cmd, pipeline, obj_allocator);
        entry_point.get()->bind_entry_point_parameter("params", params, cmd, pipeline,
                                                      obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto result = readback(output_buffer, 2);
    EXPECT_EQ(result.data[0], float_bits(1.5f));
    EXPECT_EQ(result.data[1], int_bits(-7));
}

// ===========================================================================
// 11. Shared sub-objects
// ===========================================================================

TEST_F(SlangBindingTest, SharedCBacrossPBs) {
    // Same CB sub-object assigned to two different ParameterBlocks.
    // Both PBs must get the CB's buffer descriptor in their descriptor sets.
    auto ctx = create_multi_pb_context("shared_cb", 2);

    auto output_a = ctx.output_buffer;
    auto output_b = allocator->create_buffer(
        2 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output_b");

    // Create PB objects for 'a' and 'b'
    auto a_obj = ctx.entry_point->create_shader_object(context, "a", allocator);
    auto b_obj = ctx.entry_point->create_shader_object(context, "b", allocator);

    // Create ONE shared CB and write values to it
    auto shared_cb = a_obj->create_subobject("cb");
    {
        auto c = shared_cb->get_cursor();
        c["x"] = 7.5f;
        c["y"] = -42;
    }

    // Assign the same CB to both PBs
    a_obj->set_subobject("cb", shared_cb);
    b_obj->set_subobject("cb", shared_cb);

    // Set outputs
    a_obj->get_cursor()["output"] = output_a;
    b_obj->get_cursor()["output"] = output_b;

    // Dispatch
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(ctx.pipeline);
        ctx.entry_point->bind_entry_point_parameter("a", a_obj, cmd, ctx.pipeline,
                                                    ctx.obj_allocator);
        ctx.entry_point->bind_entry_point_parameter("b", b_obj, cmd, ctx.pipeline,
                                                    ctx.obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    // Both PBs should see the same CB values
    auto result_a = readback(output_a, 2);
    auto result_b = readback(output_b, 2);

    EXPECT_EQ(result_a.data[0], float_bits(7.5f));
    EXPECT_EQ(result_a.data[1], int_bits(-42));
    EXPECT_EQ(result_b.data[0], float_bits(7.5f));
    EXPECT_EQ(result_b.data[1], int_bits(-42));
}

// version propagation through the composition → program → entry point chain

TEST_F(SlangBindingTest, CompositionChangePropagatesToProgram) {
    auto composition = SlangComposition::create();
    composition->add_module_from_string("constants",
                                        "namespace test { export static const int N = 1; }");

    auto program = SlangProgram::create(compile_context, composition);
    program.get();
    auto v0 = program.version();

    // Modify the composition in-place (same module name, different content)
    composition->add_module_from_string("constants",
                                        "namespace test { export static const int N = 2; }");

    program.get(); // pull rebuilds the dirtied program
    EXPECT_GT(program.version(), v0) << "Program version should increment when composition changes";
}

TEST_F(SlangBindingTest, CompositionChangePropagatesToEntryPoint) {
    auto composition = SlangComposition::create();
    composition->add_module_from_string("constants",
                                        "namespace test { export static const int N = 1; }");
    composition->add_module_from_string(
        "shader", "[shader(\"compute\")][numthreads(1,1,1)] void main() {}", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    entry_point.get();
    auto v0 = entry_point.version();

    composition->add_module_from_string("constants",
                                        "namespace test { export static const int N = 2; }");

    entry_point.get(); // pull rebuilds the dirtied entry point
    EXPECT_GT(entry_point.version(), v0)
        << "EntryPoint version should increment when composition changes";
}

TEST_F(SlangBindingTest, SubCompositionChangePropagatesToParent) {
    auto sub = SlangComposition::create();
    sub->add_module_from_string("sub_const", "namespace sub { export static const int X = 1; }");

    auto parent = SlangComposition::create();
    parent->add_composition(sub);
    auto v0 = parent->version();

    sub->add_module_from_string("sub_const", "namespace sub { export static const int X = 2; }");

    EXPECT_GT(parent->version(), v0)
        << "Parent composition version should increment when sub-composition changes";
}

TEST_F(SlangBindingTest, ForceReloadTriggersFullChain) {
    auto composition = SlangComposition::create();
    composition->add_module_from_string(
        "shader", "[shader(\"compute\")][numthreads(1,1,1)] void main() {}", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");
    entry_point.get(); // build the chain

    auto comp_v0 = composition->version();
    auto prog_v0 = program.version();
    auto ep_v0 = entry_point.version();

    composition->force_reload();
    entry_point.get(); // pull rebuilds program + entry point

    EXPECT_GT(composition->version(), comp_v0);
    EXPECT_GT(program.version(), prog_v0);
    EXPECT_GT(entry_point.version(), ep_v0);
}
