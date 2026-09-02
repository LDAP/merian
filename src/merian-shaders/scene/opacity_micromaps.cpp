#include "merian-shaders/scene/opacity_micromaps.hpp"

#include "merian/utils/alignment.hpp"
#include "merian/utils/properties.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/utils/profiler.hpp"

namespace merian {

namespace {
constexpr uint32_t BAKE_GROUP_SIZE = 64;
constexpr uint32_t STATES_PER_WORD = 16;
// device addresses of the micromap build inputs must be a multiple of this
constexpr vk::DeviceSize INPUT_ALIGNMENT = MicromapBuilder::INPUT_ALIGNMENT;
// Worst-case alpha samples one bake dispatch may take. The classification stops a micro-triangle
// as soon as two samples disagree, so the real count is far lower; this only has to stay under
// what the driver's reset timeout tolerates.
constexpr uint64_t MAX_BAKE_SAMPLES = 25'000'000'000;

uint32_t words_per_triangle(const uint32_t subdivision_level) {
    return std::max(1u, (1u << (2 * subdivision_level)) / STATES_PER_WORD);
}
} // namespace

OpacityMicromaps::OpacityMicromaps(const ShaderCompileContextHandle& compile_context,
                                   const ContextHandle& context,
                                   const ResourceAllocatorHandle& allocator)
    : compile_context(compile_context), context(context), allocator(allocator),
      builder(context, allocator) {}

bool OpacityMicromaps::is_supported(const ContextHandle& context) {
    return MicromapBuilder::is_supported(context);
}

std::vector<std::string> OpacityMicromaps::optional_device_features() {
    return {"micromap"};
}

uint32_t OpacityMicromaps::max_subdivision_level() const {
    return context->get_physical_device()
        ->get_properties()
        .get_opacity_micromap_properties_ext()
        .maxOpacity4StateSubdivisionLevel;
}

void OpacityMicromaps::ensure_pipeline(const SlangCompositionHandle& material_system_composition) {
    if (composition) {
        return;
    }
    composition = SlangComposition::create();
    composition->add_composition(material_system_composition);
    composition->add_module_from_path("merian-shaders/scene/omm-bake.slang", true);
    program = SlangProgram::create(compile_context, composition);
    entry_point = SlangProgramEntryPoint::create(program, "main");
    pipeline = Versioned<Pipeline>([this] {
        const auto ep = entry_point.get();
        return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
    });
    pipeline.depends_on(entry_point);
    params = Versioned<ShaderObject>([this] {
        return entry_point->create_shader_object_for_parameter(context, "params", allocator);
    });
    params.depends_on(entry_point);
}

void OpacityMicromaps::build(const CommandBufferHandle& cmd,
                             const std::vector<MeshGeometry>& meshes,
                             const MaterialSystemHandle& material_system,
                             const uint32_t subdivision_level,
                             const uint32_t max_samples_per_edge,
                             const ShaderObjectAllocatorHandle& obj_allocator_in) {
    clear();
    if (meshes.empty()) {
        return;
    }
    MERIAN_PROFILE_SCOPE_GPU(cmd, "OpacityMicromaps::build");

    ensure_pipeline(material_system->get_composition());

    ShaderObjectAllocatorHandle obj_allocator = obj_allocator_in;
    if (!obj_allocator) {
        if (!fallback_obj_allocator)
            fallback_obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);
        obj_allocator = fallback_obj_allocator;
    }

    // 1. keep the bake inside one dispatch the device is willing to finish: the budget is a
    // worst case per micro-triangle, so lower it until the whole bake fits the sample cap.
    const uint32_t micro_triangles = 1u << (2 * subdivision_level);
    uint64_t micro_triangle_count = 0;
    for (const MeshGeometry& mesh : meshes) {
        micro_triangle_count += uint64_t(mesh.geometry.primitive_count) * micro_triangles;
    }
    uint32_t budget = std::max(1u, max_samples_per_edge);
    const auto samples_per_micro_triangle = [](const uint32_t edge) {
        return uint64_t(edge + 1) * (edge + 2) / 2;
    };
    while (budget > 1 &&
           micro_triangle_count * samples_per_micro_triangle(budget) > MAX_BAKE_SAMPLES) {
        budget--;
    }
    if (budget != max_samples_per_edge) {
        SPDLOG_WARN("opacity micromaps: {} micro-triangles, lowering the sample budget {} -> {}",
                    micro_triangle_count, max_samples_per_edge, budget);
    }

    // 2. lay the meshes out in the shared buffers, each micromap's inputs aligned for the build
    const uint32_t tri_words = words_per_triangle(subdivision_level);
    const uint32_t data_word_alignment = INPUT_ALIGNMENT / sizeof(uint32_t);
    const uint32_t triangle_alignment = INPUT_ALIGNMENT / sizeof(OmmTriangle);

    std::vector<OmmBakeJob> jobs;
    jobs.reserve(meshes.size());
    uint32_t word_offset = 0;
    uint32_t data_word_offset = 0;
    uint32_t triangle_offset = 0;
    for (const MeshGeometry& mesh : meshes) {
        OmmBakeJob job{};
        job.geometry = mesh.geometry;
        job.subdivision_level = subdivision_level;
        job.max_samples_per_edge = budget;
        job.word_offset = word_offset;
        job.data_word_offset = data_word_offset;
        job.triangle_offset = triangle_offset;
        jobs.emplace_back(job);

        const uint32_t words = mesh.geometry.primitive_count * tri_words;
        word_offset += words;
        data_word_offset = align_ceil(data_word_offset + words, data_word_alignment);
        triangle_offset =
            align_ceil(triangle_offset + mesh.geometry.primitive_count, triangle_alignment);
    }

    const uint32_t word_count = word_offset;
    data_size = static_cast<vk::DeviceSize>(data_word_offset) * sizeof(uint32_t);
    triangle_count = triangle_offset;

    // 3. (re)allocate the shared buffers
    const auto ensure = [&](BufferHandle& buffer, const vk::DeviceSize size,
                            const vk::BufferUsageFlags usage, const std::string& name) {
        if (buffer && buffer->get_size() >= size) {
            return;
        }
        if (buffer) {
            cmd->keep_until_pool_reset(std::move(buffer));
        }
        buffer = allocator->create_buffer(std::max<vk::DeviceSize>(size, INPUT_ALIGNMENT), usage,
                                          MemoryMappingType::NONE, name, INPUT_ALIGNMENT);
    };
    const vk::BufferUsageFlags build_input =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
        vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT;
    ensure(data_buffer, data_size, build_input, "micromap data");
    ensure(triangle_buffer, static_cast<vk::DeviceSize>(triangle_count) * sizeof(OmmTriangle),
           build_input, "micromap triangles");

    if (job_buffer) {
        cmd->keep_until_pool_reset(std::move(job_buffer));
    }
    job_buffer = allocator->create_buffer(
        cmd, jobs.size() * sizeof(OmmBakeJob),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        jobs.data(), MemoryMappingType::NONE, "micromap bake jobs");

    // 4. bake the opacity states
    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "bake");
        const auto ep = entry_point.get();
        const auto pipe = pipeline.get();
        const ShaderObjectHandle param_object = params.get();
        auto c = param_object->get_cursor();
        c["jobs"] = job_buffer;
        c["data"] = data_buffer;
        c["triangles"] = triangle_buffer;
        c["job_count"] = static_cast<uint32_t>(jobs.size());
        c["word_count"] = word_count;

        cmd->bind(pipe);
        ep->bind("material_system", material_system->get_shader_object(), cmd, pipe, obj_allocator);
        ep->bind("params", param_object, cmd, pipe, obj_allocator);
        cmd->dispatch((word_count + BAKE_GROUP_SIZE - 1) / BAKE_GROUP_SIZE, 1, 1);
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
        vk::PipelineStageFlagBits2::eMicromapBuildEXT, vk::AccessFlagBits2::eMicromapReadEXT});

    // 5. build one micromap per mesh over its slice of the shared buffers
    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "build");
        for (uint32_t i = 0; i < meshes.size(); i++) {
            const vk::MicromapUsageEXT usage{meshes[i].geometry.primitive_count, subdivision_level,
                                             MERIAN_OMM_FORMAT_4_STATE};
            const MicromapHandle micromap = builder.queue_build(
                usage, data_buffer, static_cast<vk::DeviceSize>(jobs[i].data_word_offset) * 4,
                triangle_buffer,
                static_cast<vk::DeviceSize>(jobs[i].triangle_offset) * sizeof(OmmTriangle),
                "micromap");
            entries.try_emplace(meshes[i].mesh_id, Entry{micromap, usage});
        }
        builder.get_cmds(cmd);
    }
}

const MicromapHandle& OpacityMicromaps::get(const uint32_t mesh_id) const {
    static const MicromapHandle none;
    const auto it = entries.find(mesh_id);
    return it == entries.end() ? none : it->second.micromap;
}

const vk::MicromapUsageEXT& OpacityMicromaps::get_usage(const uint32_t mesh_id) const {
    static const vk::MicromapUsageEXT none;
    const auto it = entries.find(mesh_id);
    return it == entries.end() ? none : it->second.usage;
}

void OpacityMicromaps::clear() {
    entries.clear();
    data_size = 0;
    triangle_count = 0;
}

void OpacityMicromaps::properties(Properties& props) {
    props.output_text(fmt::format("micromaps: {}\ndata: {}\ntriangles: {}", entries.size(),
                                  format_size(data_size), triangle_count));
}

} // namespace merian
