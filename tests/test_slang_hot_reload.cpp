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
#include <filesystem>
#include <fstream>
#include <random>

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
            // Materials store fp16 (half3 tint/emission), so the shaders declare 16-bit
            // StorageBuffer variables; without these the validation layer rejects them.
            .features = VulkanFeatures(
                {"scalarBlockLayout", "shaderInt64", "shaderInt16", "shaderFloat16",
                 "storageBuffer16BitAccess", "uniformAndStorageBuffer16BitAccess",
                 "storageBuffer8BitAccess", "uniformAndStorageBuffer8BitAccess",
                 "shaderSampledImageArrayNonUniformIndexing", "runtimeDescriptorArray"}),
            .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
            .application_name = "test-slang-hot-reload",
        };
        context = Context::create(info);
        auto resources = context->get_context_extension<ExtensionResources>();
        allocator = resources->resource_allocator();
        queue = context->get_queue(vk::QueueFlagBits::eGraphics);
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

    static uint32_t
    dispatch_output(const SlangCompositionHandle& composition,
                    const ShaderCompileContextHandle& shader_compile_context = compile_context) {
        const auto program = SlangProgram::create(shader_compile_context, composition);
        const auto entry_point = SlangProgramEntryPoint::create(program, "main");
        const auto pipeline = ComputePipeline::create(
            entry_point.get()->get_pipeline_layout(context), entry_point.get()->specialize());

        const auto output_buffer = allocator->create_buffer(
            sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::HOST_ACCESS_RANDOM, "dispatch_output");
        const auto params =
            entry_point.get()->create_shader_object_for_parameter(context, "params", allocator);
        params->get_cursor()["output"] = output_buffer;
        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            cmd->bind(pipeline);
            entry_point.get()->bind("params", params, cmd, pipeline, obj_allocator);
            cmd->dispatch(1, 1, 1);
        });
        const uint32_t result = output_buffer->get_memory()->map_as<uint32_t>()[0];
        output_buffer->get_memory()->unmap();
        return result;
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
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 16);
    auto ms = std::make_shared<MaterialSystem>(compile_context, context, allocator, tm);

    auto type_id = ms->register_material_type(
        "merian::DiffuseMaterial", "merian-shaders/shading/materials/diffuse-material.slang");

    DiffuseMaterial mat1(float4(0.25f, 0.5f, 0.75f, 1.0f), TextureID(-1));
    MaterialID mat1_id = ms->add_material(type_id, mat1);

    // Build the pipeline
    auto composition = SlangComposition::create();
    composition->add_composition(ms->get_composition());
    composition->add_module_from_path("material_system/read_material.slang", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");

    // The pipeline is a derived node: it rebuilds itself whenever the entry point (composition)
    // changes, pulled lazily with get().
    uint32_t rebuild_count = 0;
    auto pipeline = Versioned<Pipeline>([&]() -> std::shared_ptr<Pipeline> {
        const auto ep = entry_point.get();
        rebuild_count++;
        return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
    });
    pipeline.depends_on(entry_point);
    pipeline.get(); // initial build

    // Run first dispatch
    {
        auto output_buffer = allocator->create_buffer(
            7 * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::HOST_ACCESS_RANDOM, "test_output_1");

        auto params =
            entry_point.get()->create_shader_object_for_parameter(context, "params", allocator);
        params->get_cursor()["ms"] = ms;
        params->get_cursor()["output"] = output_buffer;
        params->get_cursor()["material_id"] = static_cast<uint32_t>(mat1_id);

        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            ms->update(cmd);
            cmd->bind(pipeline.get());
            entry_point.get()->bind("params", params, cmd, pipeline.get(), obj_allocator);
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

    // Force reload — pulling the pipeline rebuilds it because the entry point changed.
    EXPECT_EQ(rebuild_count, 1u);
    composition->force_reload();
    EXPECT_EQ(rebuild_count, 1u) << "force_reload only marks dirty, no eager rebuild";
    pipeline.get();
    EXPECT_EQ(rebuild_count, 2u) << "pulling the pipeline rebuilt it after the reload";

    // Add a second material and dispatch with the auto-rebuilt pipeline
    DiffuseMaterial mat2(float4(0.1f, 0.2f, 0.3f, 0.4f), TextureID(-1));
    MaterialID mat2_id = ms->add_material(type_id, mat2);

    {
        auto output_buffer = allocator->create_buffer(
            7 * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            MemoryMappingType::HOST_ACCESS_RANDOM, "test_output_2");

        auto params =
            entry_point.get()->create_shader_object_for_parameter(context, "params", allocator);
        params->get_cursor()["ms"] = ms;
        params->get_cursor()["output"] = output_buffer;
        params->get_cursor()["material_id"] = static_cast<uint32_t>(mat2_id);

        queue->submit_wait([&](const CommandBufferHandle& cmd) {
            ms->update(cmd);
            cmd->bind(pipeline.get());
            entry_point.get()->bind("params", params, cmd, pipeline.get(), obj_allocator);
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
// Generated-source nodes (Reduce, Shadertoy) recompile a module under a fixed name with changed
// content.
// ---------------------------------------------------------------------------

namespace {

SlangCompositionHandle composition_writing(const std::string& module_name, const uint32_t value) {
    const auto composition = SlangComposition::create();
    composition->add_module_from_string(module_name,
                                        fmt::format(R"(
            struct Params {{ RWStructuredBuffer<uint> output; }};
            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(ParameterBlock<Params> params) {{ params.output[0] = {}u; }}
        )",
                                                    value),
                                        true);
    return composition;
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
}

std::filesystem::path create_temp_directory(const std::string& prefix) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                      fmt::format("{}-{:08x}", prefix, std::random_device{}());
    std::filesystem::create_directories(dir);
    return dir;
}

} // namespace

TEST_F(SlangHotReloadTest, RegeneratedModuleUnderFixedNamePicksUpNewSource) {
    // Distinct names never collide: two Reduce instances must not shadow one another.
    EXPECT_EQ(dispatch_output(composition_writing("regen_a", 7)), 7u);
    EXPECT_EQ(dispatch_output(composition_writing("regen_b", 8)), 8u);

    EXPECT_EQ(dispatch_output(composition_writing("regen_fixed_name", 11)), 11u);
    EXPECT_EQ(dispatch_output(composition_writing("regen_fixed_name", 22)), 22u);
    EXPECT_EQ(dispatch_output(composition_writing("regen_fixed_name", 33)), 33u);
    // back to a source the session already holds
    EXPECT_EQ(dispatch_output(composition_writing("regen_fixed_name", 11)), 11u);
}

TEST_F(SlangHotReloadTest, LiveCompositionsShareANameWithDifferentSources) {
    const auto first = composition_writing("shared_name", 1);
    const auto second = composition_writing("shared_name", 2);

    EXPECT_EQ(dispatch_output(first), 1u);
    EXPECT_EQ(dispatch_output(second), 2u);
    EXPECT_EQ(dispatch_output(first), 1u);
}

TEST_F(SlangHotReloadTest, OneSourceUnderTwoNames) {
    EXPECT_EQ(dispatch_output(composition_writing("twin_a", 42)), 42u);
    EXPECT_EQ(dispatch_output(composition_writing("twin_b", 42)), 42u);
}

TEST_F(SlangHotReloadTest, ManyRebindsOfOneNameKeepPickingUpNewSource) {
    for (uint32_t value = 1; value <= 80; value++) {
        ASSERT_EQ(dispatch_output(composition_writing("rebound_often", value)), value);
    }
}

TEST_F(SlangHotReloadTest, SourceStringUnderAnExistingPathRunsItsOwnSource) {
    const std::filesystem::path dir = create_temp_directory("merian-path-test");
    const ShaderCompileContextHandle path_context = ShaderCompileContext::create(context);
    path_context->add_search_path(dir);

    const auto writing = [](const uint32_t value) {
        return fmt::format(
            "struct Params {{ RWStructuredBuffer<uint> output; }};\n"
            "[shader(\"compute\")]\n"
            "[numthreads(1, 1, 1)]\n"
            "void main(ParameterBlock<Params> params) {{ params.output[0] = {}u; }}\n",
            value);
    };
    write_file(dir / "on_disk.slang", writing(1));
    const auto from_file = SlangComposition::create();
    from_file->add_module_from_path("on_disk.slang", true);
    EXPECT_EQ(dispatch_output(from_file, path_context), 1u);

    const auto from_string = SlangComposition::create();
    from_string->add_module(SlangComposition::SlangModule::from_source(
        "named_like_a_file", writing(2), "on_disk.slang", true));
    EXPECT_EQ(dispatch_output(from_string, path_context), 2u);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST_F(SlangHotReloadTest, OneNameWithTwoSourcesInOneCompositionFails) {
    const auto nested = SlangComposition::create();
    nested->add_module_from_string("one_name",
                                   "namespace one { export static const uint value = 1u; }");
    const auto composition = composition_writing("one_name_reader", 1);
    composition->add_composition(nested);
    composition->add_module_from_string("one_name",
                                        "namespace one { export static const uint value = 2u; }");
    EXPECT_THROW(SlangProgram::create(compile_context, composition).get(),
                 ShaderCompiler::compilation_failed);
}

TEST_F(SlangHotReloadTest, RebindingAnImportedModulePicksUpNewSource) {
    const auto composition_reading = [](const uint32_t value) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_string(
            "imported_config",
            fmt::format("module imported_config; public static const uint config_value = {}u;",
                        value));
        composition->add_module_from_string("config_reader", R"(
            import imported_config;
            struct Params { RWStructuredBuffer<uint> output; };
            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(ParameterBlock<Params> params) { params.output[0] = config_value; }
        )",
                                            true);
        return composition;
    };

    EXPECT_EQ(dispatch_output(composition_reading(3)), 3u);
    EXPECT_EQ(dispatch_output(composition_reading(4)), 4u);
    EXPECT_EQ(dispatch_output(composition_reading(3)), 3u);
}

// ---------------------------------------------------------------------------
// On-disk cache: every source a shader depends on reaches it, across launches and within one.
// ---------------------------------------------------------------------------

namespace {

struct CacheSources {
    uint32_t generated = 1;
    uint32_t leaf = 1;
    uint32_t library = 1;
    uint32_t middle = 1;
    uint32_t entry = 1;

    uint32_t expected() const {
        return (generated * 10000) + (leaf * 1000) + (library * 100) + (middle * 10) + entry;
    }
};

void write_sources(const std::filesystem::path& dir, const CacheSources& sources) {
    write_file(dir / "leaf.slangh", fmt::format("static const uint LEAF = {}u;\n", sources.leaf));
    write_file(dir / "library.slang",
               fmt::format("#include \"leaf.slangh\"\n"
                           "public static const uint LIBRARY_LEAF = LEAF * 1000u;\n"
                           "public uint library_value() {{ return {}u * 100u; }}\n",
                           sources.library));
    write_file(
        dir / "middle.slang",
        fmt::format(
            "import library;\n"
            "public uint middle_value() {{ return LIBRARY_LEAF + library_value() + {}u * 10u; }}\n",
            sources.middle));
    write_file(
        dir / "entry.slang",
        fmt::format("import middle;\n"
                    "import cache_test_generated;\n"
                    "struct Params {{ RWStructuredBuffer<uint> output; }};\n"
                    "[shader(\"compute\")]\n"
                    "[numthreads(1, 1, 1)]\n"
                    "void main(ParameterBlock<Params> params) {{\n"
                    "    params.output[0] = Generated.value * 10000u + middle_value() + {}u;\n"
                    "}}\n",
                    sources.entry));
}

SlangCompositionHandle cache_test_composition(const CacheSources& sources) {
    const auto composition = SlangComposition::create();
    // importers bake in the typealias target
    std::string generated = "module cache_test_generated;\n";
    for (uint32_t digit = 1; digit < 10; digit++) {
        generated += fmt::format(
            "public struct Generated{0} {{ public static const uint value = {0}u; }}\n", digit);
    }
    generated += fmt::format("public typealias Generated = Generated{};\n", sources.generated);
    composition->add_module_from_string("cache_test_generated", generated);
    composition->add_module_from_path("middle.slang");
    composition->add_module_from_path("entry.slang", true);
    return composition;
}

} // namespace

TEST_F(SlangHotReloadTest, CachedShadersFollowEverySourceEdit) {
    const std::filesystem::path dir = create_temp_directory("merian-cache-test");

    // a new session over the same on-disk cache
    const auto launch = [&](const CacheSources& sources) {
        write_sources(dir, sources);
        const ShaderCompileContextHandle context_for_launch = ShaderCompileContext::create(context);
        context_for_launch->add_search_path(dir);
        return dispatch_output(cache_test_composition(sources), context_for_launch);
    };

    CacheSources sources;
    EXPECT_EQ(launch(sources), sources.expected()) << "cold";
    EXPECT_EQ(launch(sources), sources.expected()) << "unchanged, from cache";

    sources.leaf = 2;
    EXPECT_EQ(launch(sources), sources.expected()) << "header included by an imported library";
    sources.library = 3;
    EXPECT_EQ(launch(sources), sources.expected()) << "imported library";
    sources.middle = 4;
    EXPECT_EQ(launch(sources), sources.expected()) << "composition module";
    sources.entry = 5;
    EXPECT_EQ(launch(sources), sources.expected()) << "entry point module";
    sources.generated = 6;
    EXPECT_EQ(launch(sources), sources.expected()) << "generated module";

    // back to sources cached earlier
    sources = {};
    EXPECT_EQ(launch(sources), sources.expected()) << "reverted to the first sources";
    sources.leaf = 2;
    EXPECT_EQ(launch(sources), sources.expected()) << "reverted to a cached header";

    // within one launch
    write_sources(dir, sources);
    const ShaderCompileContextHandle live_context = ShaderCompileContext::create(context);
    live_context->add_search_path(dir);
    const SlangCompositionHandle live_composition = cache_test_composition(sources);
    EXPECT_EQ(dispatch_output(live_composition, live_context), sources.expected()) << "live";
    sources.leaf = 7;
    sources.library = 8;
    write_sources(dir, sources);
    live_composition->force_reload();
    EXPECT_EQ(dispatch_output(live_composition, live_context), sources.expected())
        << "live, header and library edited";
    sources.middle = 9;
    write_sources(dir, sources);
    EXPECT_TRUE(live_composition->reload(live_context->get_search_path_file_loader()));
    EXPECT_EQ(dispatch_output(live_composition, live_context), sources.expected())
        << "live, composition module edited";

    // a module first compiled after an unreloaded edit
    const auto library_only = [&](const ShaderCompileContextHandle& shader_compile_context) {
        write_file(dir / "entry_b.slang", "import library;\n"
                                          "struct Params { RWStructuredBuffer<uint> output; };\n"
                                          "[shader(\"compute\")]\n"
                                          "[numthreads(1, 1, 1)]\n"
                                          "void main(ParameterBlock<Params> params) {\n"
                                          "    params.output[0] = LIBRARY_LEAF + library_value();\n"
                                          "}\n");
        const auto composition = SlangComposition::create();
        composition->add_module_from_path("entry_b.slang", true);
        return dispatch_output(composition, shader_compile_context);
    };
    sources.leaf = 3;
    write_sources(dir, sources);
    // a session keeps what it read until reloaded, and caches nothing under the new content
    library_only(live_context);
    live_composition->force_reload();
    EXPECT_EQ(library_only(live_context), (sources.leaf * 1000) + (sources.library * 100))
        << "live, header edited, after reload";
    const ShaderCompileContextHandle next_context = ShaderCompileContext::create(context);
    next_context->add_search_path(dir);
    EXPECT_EQ(library_only(next_context), (sources.leaf * 1000) + (sources.library * 100))
        << "next launch after an unreloaded edit";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST_F(SlangHotReloadTest, CachedModulesSurviveAnImportUnderAnotherSpelling) {
    if (const char* cache = std::getenv("MERIAN_SHADER_CACHE");
        cache != nullptr && std::string_view{cache} == "0") {
        GTEST_SKIP() << "shader cache disabled";
    }
    const std::filesystem::path dir = create_temp_directory("merian-spelling-test");
    std::filesystem::create_directories(dir / "lib");
    write_file(dir / "lib" / "rng.slang", "public struct Rng { public uint state; }\n");
    write_file(dir / "lib" / "dist.slang", "__exported import lib.rng;\n"
                                           "public interface IDist {\n"
                                           "    uint sample(inout Rng rng);\n"
                                           "    uint sample_twice(inout Rng rng);\n"
                                           "}\n");
    write_file(dir / "lib" / "impl.slang",
               "__exported import lib.dist;\n"
               "public struct Impl : IDist {\n"
               "    public uint sample(inout Rng rng) { return 7u; }\n"
               "    public uint sample_twice(inout Rng rng) { return 8u; }\n"
               "}\n");
    write_file(dir / "entry.slang", "import lib.impl;\n"
                                    "struct Params { RWStructuredBuffer<uint> output; };\n"
                                    "[shader(\"compute\")]\n"
                                    "[numthreads(1, 1, 1)]\n"
                                    "void main(ParameterBlock<Params> params) {\n"
                                    "    Rng rng;\n"
                                    "    rng.state = 0u;\n"
                                    "    Impl impl;\n"
                                    "    params.output[0] = impl.sample(rng) * 10u + "
                                    "impl.sample_twice(rng);\n"
                                    "}\n");

    const auto launch = [&](const std::optional<std::string>& first_import) {
        const ShaderCompileContextHandle context_for_launch = ShaderCompileContext::create(context);
        context_for_launch->add_search_path(dir);
        const auto composition = SlangComposition::create();
        if (first_import) {
            composition->add_module_from_string(
                "imports_first",
                fmt::format("import {};\npublic uint unused() {{ return 0u; }}\n", *first_import));
        }
        composition->add_module_from_path("entry.slang", true);
        return dispatch_output(composition, context_for_launch);
    };

    // as if the size cap evicted it
    const auto evict_cached_module = [](const std::filesystem::path& source) {
        const char* cache_dir = std::getenv("MERIAN_SHADER_CACHE_DIR");
        const std::filesystem::path root =
            cache_dir != nullptr ? cache_dir : std::filesystem::current_path() / ".merian-cache";
        const std::string file_name = fmt::format(
            "{:016x}.slang-module", hash_val(std::filesystem::weakly_canonical(source).string()));
        uint32_t evicted = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.path().filename() == file_name) {
                evicted += std::filesystem::remove(entry.path()) ? 1 : 0;
            }
        }
        return evicted;
    };

    EXPECT_EQ(launch(std::nullopt), 78u) << "cold";
    ASSERT_GT(evict_cached_module(dir / "lib" / "rng.slang"), 0u);
    ASSERT_GT(evict_cached_module(dir / "lib" / "dist.slang"), 0u);
    // `rng` compiles named after this import, so `dist` mangles its requirements differently than
    // the cached `impl` refers to them
    EXPECT_EQ(launch("\"lib/rng.slang\""), 78u) << "dependency compiled under another name";
    EXPECT_EQ(launch(std::nullopt), 78u) << "cache intact";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST_F(SlangHotReloadTest, ModuleOutsideTheSearchPathsFollowsEdits) {
    const std::filesystem::path dir = create_temp_directory("merian-outside-test");
    const std::filesystem::path module = dir / "outside.slang";

    const auto launch = [&](const uint32_t value) {
        write_file(module, fmt::format("struct Params {{ RWStructuredBuffer<uint> output; }};\n"
                                       "[shader(\"compute\")]\n"
                                       "[numthreads(1, 1, 1)]\n"
                                       "void main(ParameterBlock<Params> params) {{\n"
                                       "    params.output[0] = {}u;\n"
                                       "}}\n",
                                       value));
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        return dispatch_output(composition, ShaderCompileContext::create(context));
    };

    EXPECT_EQ(launch(1), 1u);
    EXPECT_EQ(launch(2), 2u) << "edited";
    EXPECT_EQ(launch(1), 1u) << "reverted";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// ---------------------------------------------------------------------------
// End-to-end: texture manager array resize → pipeline rebuild → GPU read
// ---------------------------------------------------------------------------

TEST_F(SlangHotReloadTest, TextureArrayResizeRebuildsFullPipeline) {
    // Start with capacity 4
    auto tm = std::make_shared<TextureManager>(compile_context, context, allocator, 4);

    // Build pipeline using TM composition
    const auto composition = SlangComposition::create();
    composition->add_composition(tm->get_composition());
    composition->add_module_from_path("texture_manager/read_texture.slang", true);

    auto program = SlangProgram::create(compile_context, composition);
    auto entry_point = SlangProgramEntryPoint::create(program, "main");

    // The pipeline rebuilds itself when the entry point (TM composition) changes; pulled with
    // get().
    uint32_t rebuild_count = 0;
    auto pipeline = Versioned<Pipeline>([&]() -> std::shared_ptr<Pipeline> {
        const auto ep = entry_point.get();
        rebuild_count++;
        return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
    });
    pipeline.depends_on(entry_point);
    pipeline.get(); // initial build

    // Fill to capacity
    auto dummy = allocator->get_dummy_texture();
    for (uint32_t i = 0; i < 4; i++) {
        tm->add_texture(dummy);
    }
    pipeline.get();
    EXPECT_EQ(rebuild_count, 1u) << "No resize yet, pipeline unchanged";

    // Add a real texture — triggers resize 4 → 8 → pipeline rebuilds on next pull
    const uint32_t red_pixel = 0xFF0000FF;
    TextureID tex_id;
    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        tex_id = tm->add_texture_from_rgba8(cmd, &red_pixel, 1, 1, vk::SamplerAddressMode::eRepeat,
                                            vk::Filter::eNearest, vk::Filter::eNearest, false);
    });
    EXPECT_EQ(tm->get_capacity(), 8u);
    pipeline.get();
    EXPECT_EQ(rebuild_count, 2u) << "pipeline rebuilt after TM resize";

    // Dispatch with the auto-rebuilt pipeline
    auto output_buffer = allocator->create_buffer(
        4 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::HOST_ACCESS_RANDOM, "test_output");

    auto params =
        entry_point.get()->create_shader_object_for_parameter(context, "params", allocator);
    params->get_cursor()["tm"] = tm;
    params->get_cursor()["output"] = output_buffer;
    params->get_cursor()["texture_id"] = static_cast<uint32_t>(tex_id);

    queue->submit_wait([&](const CommandBufferHandle& cmd) {
        cmd->bind(pipeline.get());
        entry_point.get()->bind("params", params, cmd, pipeline.get(), obj_allocator);
        cmd->dispatch(1, 1, 1);
    });

    auto* mapped = output_buffer->get_memory()->map_as<uint32_t>();
    EXPECT_NEAR(bits_float(mapped[0]), 1.0f, 1e-3f); // R
    EXPECT_NEAR(bits_float(mapped[1]), 0.0f, 1e-3f); // G
    EXPECT_NEAR(bits_float(mapped[2]), 0.0f, 1e-3f); // B
    EXPECT_NEAR(bits_float(mapped[3]), 1.0f, 1e-3f); // A
    output_buffer->get_memory()->unmap();
}
