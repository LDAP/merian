#pragma once

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

#include <memory>

namespace merian {

class Properties;
class GraphRun;

// Host side of the NVIDIA SHARC world-space radiance cache. Owns the three SHARC buffers, binds
// them and the per-frame uniforms for the render program, and drives the mandatory per-frame
// resolve compute pass. TESTING integration; see third_party/SHARC.
class Sharc {
  public:
    Sharc(const ContextHandle& context,
          const ShaderCompileContextHandle& compile_context,
          const ResourceAllocatorHandle& allocator,
          uint32_t capacity);

    static SlangCompositionHandle query_device_support_composition();

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    // SHARC requires all buffers zero-initialized; clears them and barriers for shader access.
    void reset(const CommandBufferHandle& cmd);

    // Advance per-frame state. Call once per frame before write_to()/resolve().
    void begin_frame(const float3& camera_position, uint32_t frame_index);

    // Bind buffers + per-frame uniforms into a Sharc-typed shader cursor.
    void write_to(ShaderCursor cursor) const;

    // Combine this frame's accumulation with the resolved history (compute dispatch + barriers).
    void resolve(GraphRun& run, const CommandBufferHandle& cmd);

    void properties(Properties& props);

    uint32_t get_capacity() const {
        return capacity;
    }

    // Drop the built resolve pipeline so it is rebuilt (e.g. on shader hot-reload).
    void invalidate() {
        resolve_built = false;
    }

  private:
    void ensure_resolve_built(GraphRun& run);

    static constexpr uint32_t RESOLVE_LOCAL_SIZE_X = 256;

    ContextHandle context;
    ShaderCompileContextHandle compile_context;
    ResourceAllocatorHandle allocator;

    const uint32_t capacity;

    // Tunables.
    float scene_scale = 50.0f;
    float radiance_scale = 1000.0f;
    uint32_t accumulation_frame_num = 64;
    uint32_t stale_frame_num_max = 128;

    // Per-frame uniforms.
    float3 camera_position{0};
    float3 prev_camera_position{0};
    uint32_t frame_index = 0;

    SlangCompositionHandle composition;

    BufferHandle hash_entries; // HashGridKey (uint64), 8 B
    BufferHandle accumulation; // SharcAccumulationData (uint4), 16 B
    BufferHandle resolved;     // SharcPackedData (float16_t4 + 2 u32), 16 B

    // Resolve compute pass, built lazily.
    bool resolve_built = false;
    SlangProgramEntryPointHandle resolve_entry_point;
    PipelineHandle resolve_pipeline;
    ShaderObjectHandle resolve_object;
    std::shared_ptr<FrameCachingShaderObjectAllocator> resolve_obj_allocator;
};

using SharcHandle = std::shared_ptr<Sharc>;

} // namespace merian
