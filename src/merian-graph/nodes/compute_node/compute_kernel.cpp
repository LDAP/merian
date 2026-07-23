#include "merian-graph/nodes/compute_node/compute_kernel.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

ComputeKernel::ComputeKernel(const ContextHandle& context,
                             const ResourceAllocatorHandle& allocator,
                             const ShaderCompileContextHandle& compile_context,
                             std::function<SlangCompositionHandle()> compose,
                             const Versioned<SpecializationInfo>& spec) {
    composition = Versioned<SlangComposition>(std::move(compose));
    program = SlangProgram::create(compile_context, composition);
    entry_point = SlangProgramEntryPoint::create(program, "main");

    pipeline = Versioned<Pipeline>([this, context, spec] {
        const auto ep = entry_point.get();
        return ComputePipeline::create(ep->get_pipeline_layout(context),
                                       ep->specialize(spec.get()));
    });
    pipeline.depends_on(entry_point);
    pipeline.depends_on(spec);

    globals = Versioned<ShaderObject>([this, context, allocator] {
        return entry_point.get()->create_global_shader_object(context, allocator);
    });
    globals.depends_on(entry_point);
}

ComputeKernel::ComputeKernel(const ContextHandle& context,
                             const ResourceAllocatorHandle& allocator,
                             const ShaderCompileContextHandle& compile_context,
                             const std::string& module_path,
                             const Versioned<SpecializationInfo>& spec)
    : ComputeKernel(
          context,
          allocator,
          compile_context,
          [module_path] {
              const auto composition = SlangComposition::create();
              composition->add_module_from_path(module_path, true);
              return composition;
          },
          spec) {}

PipelineHandle ComputeKernel::bind(GraphRun& run, const NodeIO& io) {
    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto global = globals.get();

    ShaderCursor cursor = global->get_cursor();
    io.bind(cursor);

    const CommandBufferHandle& cmd = run.get_cmd();
    cmd->bind(pipe);
    ep->bind_global(global, cmd, pipe, run.get_shader_object_allocator());
    return pipe;
}

ShaderCursor ComputeKernel::globals_cursor() {
    return globals.get()->get_cursor();
}

void ComputeKernel::invalidate() {
    composition.invalidate();
}

void ComputeKernel::reload(const bool force, const ShaderCompileContextHandle& compile_context) {
    const auto& current = composition.peek();
    if (!current) {
        return;
    }
    if (force) {
        current->force_reload();
    } else {
        current->reload(compile_context->get_search_path_file_loader());
    }
}

} // namespace merian
