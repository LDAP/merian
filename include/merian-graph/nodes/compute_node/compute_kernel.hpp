#pragma once

#include "merian-graph/graph/graph_run.hpp"
#include "merian-graph/graph/node_io.hpp"

#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

#include <functional>
#include <string>

namespace merian {

// One compute dispatch of a node, backed by a slang composition: the reactive shader → pipeline →
// globals chain, lazily built and hot-reloadable. The node records the dispatch.
class ComputeKernel {
  public:
    ComputeKernel(const ContextHandle& context,
                  const ResourceAllocatorHandle& allocator,
                  const ShaderCompileContextHandle& compile_context,
                  std::function<SlangCompositionHandle()> compose,
                  const Versioned<SpecializationInfo>& spec);

    ComputeKernel(const ContextHandle& context,
                  const ResourceAllocatorHandle& allocator,
                  const ShaderCompileContextHandle& compile_context,
                  const std::string& module_path,
                  const Versioned<SpecializationInfo>& spec);

    ComputeKernel(const ComputeKernel&) = delete;
    ComputeKernel& operator=(const ComputeKernel&) = delete;

    // Binds graph io; returns the pipeline so the caller can push constants and dispatch.
    PipelineHandle bind(GraphRun& run, const NodeIO& io);

    // Cursor into the global shader object, for writing node-internal (non-graph) fields before
    // bind().
    ShaderCursor globals_cursor();

    // Discards the composition so the next bind() recomposes and recompiles.
    void invalidate();

    void reload(bool force, const ShaderCompileContextHandle& compile_context);

  private:
    Versioned<SlangComposition> composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> entry_point;
    Versioned<Pipeline> pipeline;
    Versioned<ShaderObject> globals;
};

} // namespace merian
