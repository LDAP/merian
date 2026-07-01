#include "merian-graph/nodes/render_pt_mcpg/mcpg.hpp"

namespace merian {

namespace {
constexpr const char* MODULE_PATH = "merian-graph/nodes/render_pt_mcpg/mc.slang";

SlangCompositionHandle make_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(MODULE_PATH);
    return composition;
}
} // namespace

MCPG::MCPG(const ShaderCompileContextHandle& compile_context,
           const ResourceAllocatorHandle& allocator,
           const uint32_t buffer_size)
    : composition(make_composition()),
      grid(compile_context, allocator, composition, "MCState", buffer_size) {}

SlangCompositionHandle MCPG::query_device_support_composition() {
    return make_composition();
}

} // namespace merian
