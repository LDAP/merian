#include <gtest/gtest.h>

#include "merian-shaders/gbuffer.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <utility>

using namespace merian;

namespace {

using Field = GBufferField;
using Storage = GBufferLayout::Storage;

const GBufferLayout::Texture& texture_of(const GBufferLayout& layout, const Field field) {
    return layout.get_textures()[layout.get_texture_index(field)];
}

const GBufferLayout::Placement& placement_of(const GBufferLayout& layout, const Field field) {
    const GBufferLayout::Texture& texture = texture_of(layout, field);
    return *std::ranges::find(texture.placements, field, &GBufferLayout::Placement::field);
}

const std::vector<GBufferGroup> SURFACE_DENOISER = {
    {{Field::Normal, Field::LinearZ, Field::GradZ, Field::DeltaZ}},
    {{Field::MotionVectors}},
    {{Field::Albedo}},
    {{Field::Hit}},
};

const std::vector<GBufferGroup> RAY_RECONSTRUCTION = {
    {.fields = {Field::ViewDepth}, .texture = true},
    {.fields = {Field::MotionVectors}, .texture = true},
    {.fields = {Field::Normal, Field::Roughness}, .texture = true},
    {.fields = {Field::DiffuseAlbedo}, .texture = true},
    {.fields = {Field::SpecularAlbedo}, .texture = true},
    {{Field::Hit}},
};

} // namespace

TEST(GBufferLayout, FewestBytesFloatOnTies) {
    const GBufferLayout layout(SURFACE_DENOISER);

    EXPECT_FALSE(placement_of(layout, Field::Normal).packed);
    EXPECT_EQ(texture_of(layout, Field::Normal).storage, Storage::Float16);
    EXPECT_EQ(layout.get_texture_index(Field::LinearZ), layout.get_texture_index(Field::Normal));
    EXPECT_EQ(layout.get_texture_index(Field::DeltaZ), layout.get_texture_index(Field::GradZ));
    EXPECT_TRUE(placement_of(layout, Field::Hit).packed);
    EXPECT_FALSE(layout.contains(Field::ViewDepth));

    const GBufferLayout packed({{{Field::Normal, Field::GradZ}}});
    EXPECT_TRUE(placement_of(packed, Field::Normal).packed);
    EXPECT_EQ(packed.get_texture_index(Field::GradZ), packed.get_texture_index(Field::Normal));
}

TEST(GBufferLayout, TextureGroupsKeepTheirChannels) {
    const GBufferLayout layout(RAY_RECONSTRUCTION);

    for (const Field field : {Field::ViewDepth, Field::MotionVectors, Field::Normal,
                              Field::DiffuseAlbedo, Field::SpecularAlbedo}) {
        EXPECT_TRUE(texture_of(layout, field).exact);
        EXPECT_EQ(placement_of(layout, field).channel, 0u);
        EXPECT_FALSE(placement_of(layout, field).packed);
    }
    EXPECT_EQ(layout.get_texture_index(Field::Roughness), layout.get_texture_index(Field::Normal));
    EXPECT_EQ(placement_of(layout, Field::Roughness).channel, 3u);
    EXPECT_EQ(texture_of(layout, Field::ViewDepth).storage, Storage::Float32);
    EXPECT_EQ(texture_of(layout, Field::MotionVectors).channel_count, 2u);
}

TEST(GBufferLayout, ExactTexturesTakeNoOtherFields) {
    std::vector<GBufferGroup> groups = RAY_RECONSTRUCTION;
    groups.push_back({{Field::MotionVectors, Field::LinearZ}});
    const GBufferLayout layout(groups);

    EXPECT_EQ(texture_of(layout, Field::MotionVectors).channel_count, 2u);
    EXPECT_FALSE(texture_of(layout, Field::LinearZ).exact);
}

TEST(GBufferLayout, NextToTheGroupInTheFormItHas) {
    const GBufferLayout layout({
        {{Field::Normal, Field::GradZ}},
        {{Field::Normal, Field::MotionVectors}},
        {{Field::Normal, Field::LinearZ}},
        {{Field::Hit, Field::DeltaZ}},
        {{Field::DeltaZ, Field::ViewDepth}},
        {{Field::DeltaZ, Field::ProjectedDepth}},
    });
    for (const Field field : {Field::Normal, Field::GradZ, Field::MotionVectors, Field::LinearZ,
                              Field::DeltaZ, Field::ViewDepth, Field::ProjectedDepth}) {
        EXPECT_TRUE(placement_of(layout, field).packed);
    }
    EXPECT_EQ(layout.get_texture_index(Field::LinearZ), layout.get_texture_index(Field::Normal));
    EXPECT_EQ(layout.get_texture_index(Field::ViewDepth), layout.get_texture_index(Field::DeltaZ));
}

TEST(GBufferLayout, TextureGroupPrefixIsShared) {
    const GBufferLayout layout({
        {.fields = {Field::Normal}, .texture = true},
        {.fields = {Field::Normal, Field::Roughness}, .texture = true},
    });
    EXPECT_EQ(layout.get_textures().size(), 1u);
}

TEST(GBufferLayout, EmptyStillHasATexture) {
    const GBufferLayout layout({});
    EXPECT_EQ(layout.get_textures().size(), 1u);
}

TEST(GBufferLayout, EqualGroupsEqualLayouts) {
    EXPECT_TRUE(GBufferLayout(SURFACE_DENOISER) == GBufferLayout(SURFACE_DENOISER));
    EXPECT_FALSE(GBufferLayout(SURFACE_DENOISER) == GBufferLayout(RAY_RECONSTRUCTION));
}

const char* const ROUND_TRIP_SOURCE = R"(
import merian_shaders.gbuffer;
import merian_shaders.utils.encoding;

using merian;

RWStructuredBuffer<float> results;

GBufferSample make_sample() {
    GBufferSample sample;
    sample.normal = float3(0.36, -0.48, 0.8);
    sample.linear_z = 12.5;
    sample.grad_z = half2(0.25, -0.5);
    sample.delta_z = -0.75;
    sample.motion_vectors = float2(3.5, -1.25);
    sample.hit = Hit(7, 3, 11, 2, float2(0.25, 0.5));
    sample.albedo = float3(0.5, 0.25, 0.125);
    sample.diffuse_albedo = float3(0.375, 0.125, 0.0625);
    sample.specular_albedo = float3(0.125, 0.125, 0.0625);
    sample.roughness = 0.625;
    sample.view_depth = 10.25;
    sample.projected_depth = 0.875;
    return sample;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void write(uint3 tid: SV_DispatchThreadID, ParameterBlock<WGBuffer> gbuffer) {
    gbuffer.store(tid.xy, make_sample());
}

// the error of each field in GBufferField order, then of the derived accessors
[shader("compute")]
[numthreads(1, 1, 1)]
void read(uint3 tid: SV_DispatchThreadID, ParameterBlock<GBuffer> gbuffer) {
    const uint2 p = tid.xy;
    const GBufferSample expected = make_sample();
    const Hit hit = gbuffer.get_hit(p);
    const bool ids = hit.instance_id == 7 && hit.instance_index == 3 && hit.primitive_id == 11 &&
                     hit.geometry_index == 2;

    results[0] = distance(gbuffer.get_normal(p), expected.normal);
    results[1] = abs(gbuffer.get_linear_z(p) - expected.linear_z);
    results[2] = distance(float2(gbuffer.get_grad_z(p)), float2(expected.grad_z));
    results[3] = abs(gbuffer.get_delta_z(p) - expected.delta_z);
    results[4] = distance(gbuffer.get_motion_vectors(p), expected.motion_vectors);
    results[5] = ids ? distance(hit.barycentrics, expected.hit.barycentrics) : 1;
    results[6] = distance(gbuffer.get_albedo(p), expected.albedo);
    results[7] = distance(gbuffer.get_diffuse_albedo(p), expected.diffuse_albedo);
    results[8] = distance(gbuffer.get_specular_albedo(p), expected.specular_albedo);
    results[9] = abs(gbuffer.get_roughness(p) - expected.roughness);
    results[10] = abs(gbuffer.get_view_depth(p) - expected.view_depth);
    results[11] = abs(gbuffer.get_projected_depth(p) - expected.projected_depth);

    results[12] = abs(gbuffer.get_surface(p).linear_z - expected.linear_z);
    results[13] = distance(gbuffer.get_motion_vectors_dilated(int2(p), 1), expected.motion_vectors);
    results[14] = distance(decode_normal(gbuffer.get_encoded_normal(p)), expected.normal);
    results[15] = float(all(gbuffer.get_dimensions() == uint2(1)));
}
)";

class GBufferLayoutDevice : public ::testing::Test {
  protected:
    static ContextHandle context;
    static ResourceAllocatorHandle allocator;
    static QueueHandle queue;
    static ShaderCompileContextHandle compile_context;

    static void SetUpTestSuite() {
        const ContextCreateInfo info{
            .context_extensions = {ExtensionResources::name},
            .application_name = "test-gbuffer-layout",
        };
        context = Context::create(info);
        allocator = context->get_context_extension<ExtensionResources>()->resource_allocator();
        queue = context->get_queue_GCT();
        compile_context = ShaderCompileContext::create(context);
    }

    static void TearDownTestSuite() {
        compile_context.reset();
        queue.reset();
        allocator.reset();
        context.reset();
    }

    struct DeviceGBuffer {
        std::shared_ptr<GBuffer> gbuffer;
        std::vector<ImageHandle> images;
    };

    static DeviceGBuffer create_gbuffer(const GBufferLayoutHandle& layout) {
        DeviceGBuffer device_gbuffer;
        device_gbuffer.gbuffer = std::make_shared<GBuffer>(compile_context, context, allocator,
                                                           vk::Extent3D{1, 1, 1}, layout);
        std::vector<ImageViewHandle> views;
        std::vector<vk::ImageMemoryBarrier2> barriers;
        for (uint32_t i = 0; i < layout->get_textures().size(); i++) {
            const vk::ImageCreateInfo create_info{
                {},
                vk::ImageType::e2D,
                layout->get_format(i, context->get_physical_device()),
                {1, 1, 1},
                1,
                1,
                vk::SampleCountFlagBits::e1,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            };
            const ImageHandle& image = device_gbuffer.images.emplace_back(
                allocator->create_image(create_info, MemoryMappingType::NONE));
            views.emplace_back(allocator->create_image_view(image, image->make_view_create_info()));
            barriers.emplace_back(image->barrier2(vk::ImageLayout::eGeneral, true));
        }
        queue->submit_wait([&](const CommandBufferHandle& cmd) { cmd->barrier({}, {}, barriers); });
        device_gbuffer.gbuffer->set_resources(views);
        return device_gbuffer;
    }

    // Stores a sample and reads every field back through the layout.
    static std::vector<float> round_trip(const GBufferLayoutHandle& layout) {
        const DeviceGBuffer device_gbuffer = create_gbuffer(layout);

        const auto composition = SlangComposition::create();
        composition->add_composition(layout->get_composition());
        composition->add_module_from_string("gbuffer_layout_round_trip", ROUND_TRIP_SOURCE, true);
        const auto program = SlangProgram::create(compile_context, composition);
        const auto obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);

        const SlangProgramEntryPointHandle write =
            SlangProgramEntryPoint::create(program, "write").get();
        const PipelineHandle write_pipeline =
            ComputePipeline::create(write->get_pipeline_layout(context), write->specialize());
        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            cmd->bind(write_pipeline);
            write->bind("gbuffer", device_gbuffer.gbuffer->get_write_shader_object(), cmd,
                        write_pipeline, obj_allocator);
            cmd->dispatch(1, 1, 1);
        });

        constexpr uint32_t result_count = 16;
        const BufferHandle results = allocator->create_buffer(
            result_count * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
            MemoryMappingType::HOST_ACCESS_RANDOM, "results");
        const SlangProgramEntryPointHandle read =
            SlangProgramEntryPoint::create(program, "read").get();
        const PipelineHandle read_pipeline =
            ComputePipeline::create(read->get_pipeline_layout(context), read->specialize());
        const ShaderObjectHandle globals = read->create_global_shader_object(context, allocator);
        globals->get_cursor()["results"] = results;
        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            cmd->bind(read_pipeline);
            read->bind("gbuffer", device_gbuffer.gbuffer->get_shader_object(), cmd, read_pipeline,
                       obj_allocator);
            read->bind_global(globals, cmd, read_pipeline, obj_allocator);
            cmd->dispatch(1, 1, 1);
        });

        std::vector<float> errors(result_count);
        std::memcpy(errors.data(), results->get_memory()->map_as<float>(),
                    result_count * sizeof(float));
        results->get_memory()->unmap();
        return errors;
    }

    static void expect_round_trip(const GBufferLayoutHandle& layout) {
        const std::vector<float> errors = round_trip(layout);
        for (uint32_t i = 0; i < enum_size<Field>(); i++) {
            const Field field = enum_values<Field>()[i];
            if (!layout->contains(field)) {
                EXPECT_TRUE(field == Field::Hit ? errors[i] == 1.f : std::isnan(errors[i]))
                    << "unrequested field " << i << " reads " << errors[i];
            } else {
                EXPECT_LT(errors[i], 1e-3f) << "field " << i;
            }
        }
        constexpr std::array<std::pair<uint32_t, Field>, 3> DERIVED = {{
            {12, Field::LinearZ},
            {13, Field::MotionVectors},
            {14, Field::Normal},
        }};
        for (const auto& [index, field] : DERIVED) {
            if (layout->contains(field)) {
                EXPECT_LT(errors[index], 1e-3f) << "accessor " << index;
            }
        }
        EXPECT_EQ(errors[15], 1.f) << "dimensions";
    }
};

ContextHandle GBufferLayoutDevice::context;
ResourceAllocatorHandle GBufferLayoutDevice::allocator;
QueueHandle GBufferLayoutDevice::queue;
ShaderCompileContextHandle GBufferLayoutDevice::compile_context;

class GBufferLayoutRoundTrip : public GBufferLayoutDevice,
                               public ::testing::WithParamInterface<std::vector<GBufferGroup>> {};

TEST_P(GBufferLayoutRoundTrip, FieldsReadBackAsStored) {
    expect_round_trip(std::make_shared<const GBufferLayout>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    Layouts,
    GBufferLayoutRoundTrip,
    ::testing::Values(std::vector<GBufferGroup>{},
                      SURFACE_DENOISER,
                      RAY_RECONSTRUCTION,
                      std::vector<GBufferGroup>{
                          {.fields = {Field::ProjectedDepth}, .texture = true},
                          {.fields = {Field::MotionVectors}, .texture = true},
                      },
                      // every field that has a float form handed out as a texture
                      std::vector<GBufferGroup>{
                          {.fields = {Field::Normal, Field::LinearZ}, .texture = true},
                          {.fields = {Field::GradZ, Field::DeltaZ}, .texture = true},
                          {.fields = {Field::MotionVectors, Field::Roughness}, .texture = true},
                          {.fields = {Field::Albedo}, .texture = true},
                          {.fields = {Field::ViewDepth, Field::ProjectedDepth}, .texture = true},
                      },
                      // every field that has a packed form packed
                      std::vector<GBufferGroup>{
                          {{Field::Normal, Field::GradZ}},
                          {{Field::Normal, Field::MotionVectors}},
                          {{Field::Normal, Field::LinearZ}},
                          {{Field::Hit, Field::DeltaZ}},
                          {{Field::DeltaZ, Field::ViewDepth}},
                          {{Field::DeltaZ, Field::ProjectedDepth}},
                      }));

// Two gbuffers of a graph can each have their own layout.
TEST_F(GBufferLayoutDevice, LayoutsCoexist) {
    const auto surface = std::make_shared<const GBufferLayout>(SURFACE_DENOISER);
    const auto ray_reconstruction = std::make_shared<const GBufferLayout>(RAY_RECONSTRUCTION);

    expect_round_trip(surface);
    expect_round_trip(ray_reconstruction);
    expect_round_trip(surface);
}
