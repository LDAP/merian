#include "merian-graph/nodes/render_pt_mcpg/sharc.hpp"

#include "merian-graph/graph/graph_run.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/properties.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

namespace {
constexpr const char* MODULE_PATH = "merian-shaders/light-cache/sharc.slang";
constexpr const char* RESOLVE_MODULE_PATH = "merian-shaders/light-cache/sharc-resolve.slang";

// Element strides in the SHARC default (non-SH, 64-bit key) layout, in bytes.
constexpr vk::DeviceSize HASH_ENTRY_STRIDE = 8;    // HashGridKey (uint64_t)
constexpr vk::DeviceSize ACCUMULATION_STRIDE = 16; // SharcAccumulationData (uint4)
constexpr vk::DeviceSize RESOLVED_STRIDE = 16;     // SharcPackedData (float16_t4 + 2x uint32_t)

SlangCompositionHandle make_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(MODULE_PATH);
    return composition;
}
} // namespace

Sharc::Sharc(const ContextHandle& context,
             const ShaderCompileContextHandle& compile_context,
             const ResourceAllocatorHandle& allocator,
             const uint32_t capacity)
    : context(context), compile_context(compile_context), allocator(allocator), capacity(capacity),
      composition(make_composition()) {

    const auto usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    hash_entries = allocator->create_buffer(vk::DeviceSize(capacity) * HASH_ENTRY_STRIDE, usage,
                                            MemoryMappingType::NONE, "Sharc::hash_entries");
    accumulation = allocator->create_buffer(vk::DeviceSize(capacity) * ACCUMULATION_STRIDE, usage,
                                            MemoryMappingType::NONE, "Sharc::accumulation");
    resolved = allocator->create_buffer(vk::DeviceSize(capacity) * RESOLVED_STRIDE, usage,
                                        MemoryMappingType::NONE, "Sharc::resolved");
}

SlangCompositionHandle Sharc::query_device_support_composition() {
    return make_composition();
}

void Sharc::reset(const CommandBufferHandle& cmd) {
    for (const BufferHandle& buffer : {hash_entries, accumulation, resolved}) {
        cmd->fill(buffer);
        cmd->barrier(buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite));
    }
}

void Sharc::begin_frame(const float3& camera_position, const uint32_t frame_index) {
    this->prev_camera_position = this->camera_position;
    this->camera_position = camera_position;
    this->frame_index = frame_index;
}

void Sharc::write_to(ShaderCursor cursor) const {
    cursor["hash_entries"] = hash_entries;
    cursor["accumulation"] = accumulation;
    cursor["resolved"] = resolved;
    cursor["camera_position"] = camera_position;
    cursor["prev_camera_position"] = prev_camera_position;
    cursor["scene_scale"] = scene_scale;
    cursor["radiance_scale"] = radiance_scale;
    cursor["capacity"] = capacity;
    cursor["frame_index"] = frame_index;
    cursor["accumulation_frame_num"] = accumulation_frame_num;
    cursor["stale_frame_num_max"] = stale_frame_num_max;
}

void Sharc::ensure_resolve_built(GraphRun& run) {
    if (resolve_built) {
        return;
    }

    const auto resolve_composition = SlangComposition::create();
    resolve_composition->add_module_from_path(RESOLVE_MODULE_PATH, true);
    const Versioned<SlangProgram> program =
        SlangProgram::create(compile_context, resolve_composition);
    resolve_entry_point = SlangProgramEntryPoint::create(program, "main").get();

    resolve_pipeline = ComputePipeline::create(resolve_entry_point->get_pipeline_layout(context),
                                               resolve_entry_point->specialize());
    resolve_object =
        resolve_entry_point->create_shader_object_for_parameter(context, "sharc", allocator);
    resolve_obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
        allocator, run.get_iterations_in_flight());

    resolve_built = true;
}

void Sharc::resolve(GraphRun& run, const CommandBufferHandle& cmd) {
    ensure_resolve_built(run);
    resolve_obj_allocator->set_iteration(run.get_in_flight_index());

    // Render pass writes accumulation / hash entries -> resolve reads/writes them.
    std::array<vk::BufferMemoryBarrier2, 3> pre;
    const std::array<BufferHandle, 3> buffers{hash_entries, accumulation, resolved};
    for (size_t i = 0; i < buffers.size(); ++i) {
        pre[i] = buffers[i]->buffer_barrier2(
            vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }
    cmd->barrier(pre);

    auto cursor = resolve_object->get_cursor();
    write_to(cursor);

    cmd->bind(resolve_pipeline);
    resolve_entry_point->bind("sharc", resolve_object, cmd, resolve_pipeline,
                              resolve_obj_allocator);
    cmd->dispatch((capacity + RESOLVE_LOCAL_SIZE_X - 1) / RESOLVE_LOCAL_SIZE_X);

    // Resolve writes resolved / hash entries -> next frame's render pass reads them.
    std::array<vk::BufferMemoryBarrier2, 3> post;
    for (size_t i = 0; i < buffers.size(); ++i) {
        post[i] = buffers[i]->buffer_barrier2(
            vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }
    cmd->barrier(post);
}

void Sharc::properties(Properties& props) {
    props.config_float("scene scale", scene_scale,
                       "World-space voxel size control; larger means finer voxels.", 0.01f);
    props.config_float("radiance scale", radiance_scale,
                       "Quantization factor for the atomic u32 radiance accumulation. Lower for "
                       "large radiance values to avoid overflow.",
                       1.0f);
    props.config_uint("accumulation frames", accumulation_frame_num,
                      "Temporal accumulation window; larger is smoother but slower to respond.", 1u,
                      1024u);
    props.config_uint("stale frames max", stale_frame_num_max,
                      "Frames without new samples before a cache entry is evicted.", 8u, 1024u);
}

} // namespace merian
