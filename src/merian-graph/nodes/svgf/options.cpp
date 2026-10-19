#include "merian-graph/nodes/svgf/options.hpp"
#include "merian-graph/nodes/svgf/config.h"

#include "merian-shaders/gbuffer.hpp"
#include "merian/shader/spriv_reflect.hpp"
#include "merian/vk/utils/math.hpp"

namespace merian {

bool SVGFOptions::properties(Properties& config) {
    bool changed = false;
    int type = static_cast<int>(filter_type);
    if (config.config_options("type", type, {"atrous", "box", "subsampled"},
                              Properties::OptionsStyle::COMBO)) {
        changed = true;
        filter_type = static_cast<SVGFFilterType>(type);
        kaleidoscope = filter_type == SVGFFilterType::ATROUS;
    }
    changed |= config.config_bool("kaleidoscope", kaleidoscope);
    changed |= config.config_bool("kaleidoscope shmem", kaleidoscope_use_shmem,
                                  "use shared memory for kaleidoscope");
    return changed;
}

bool SVGFFilterParameters::properties(Properties& config) {
    bool changed = config.config_float("depth", depth, "more means more blur");
    float angle = std::acos(normals);
    changed |= config.config_angle("normals", angle, "Reject with normals farther apart", 0, 180);
    normals = std::cos(angle);
    changed |= config.config_float("luminance", luminance, "more means more blur", 0.1);
    changed |=
        config.config_float("z-bias normals", z_bias_normals,
                            "z-dependent rejection: increase to reject more. Disable with <= 0.");
    changed |=
        config.config_float("z-bias depth", z_bias_depth,
                            "z-dependent rejection: increase to reject more. Disable with <= 0.");
    return changed;
}

uint32_t svgf_filter_local_size(const ContextHandle& context, const SVGFOptions& options) {
    if (options.kaleidoscope && options.kaleidoscope_use_shmem) {
        return workgroup_size_for_shared_memory_with_halo(context, FILTER_SHARED_MEMORY_PER_PIXEL,
                                                          SVGF_FILTER_HALO_RADIUS,
                                                          svgf_vendor_local_size(context), 16);
    }
    return svgf_vendor_local_size(context);
}

uint32_t svgf_vendor_local_size(const ContextHandle& context) {
    return context->get_physical_device()->is_amd() ? 32 : 16;
}

SlangProgramEntryPointHandle svgf_entry_point(const ShaderCompileContextHandle& compile_context,
                                              const SlangCompositionHandle& gbuffer_composition,
                                              const std::string& module_path,
                                              const SVGFOptions& options,
                                              const vk::Format irradiance_format) {
    const std::string constants = fmt::format(
        "import merian_graph.nodes.svgf.options;\n"
        "namespace merian {{\n"
        "export static const bool svgf_variance_as_sigma = {};\n"
        "export static const SVGFFilter svgf_filter = SVGFFilter({});\n"
        "export static const bool svgf_kaleidoscope = {};\n"
        "export static const bool svgf_kaleidoscope_use_shmem = {};\n"
        "}}",
        irradiance_format == vk::Format::eR16G16B16A16Sfloat, static_cast<int>(options.filter_type),
        options.kaleidoscope, options.kaleidoscope_use_shmem);
    const auto composition = SlangComposition::create();
    composition->add_composition(gbuffer_composition);
    composition->add_module_from_path(module_path, true);
    composition->add_module_from_string("svgf_constants", constants);
    return SlangProgramEntryPoint::create(SlangProgram::create(compile_context, composition),
                                          "main")
        .get();
}

DeviceSupportInfo svgf_device_support(const DeviceSupportQueryInfo& query_info,
                                      const std::string& module_path) {
    const auto composition = SlangComposition::create();
    composition->add_composition(GBufferLayout::complete()->get_composition());
    composition->add_module_from_path(module_path, true);
    const auto binary = SlangProgramEntryPoint::create(
                            SlangProgram::create(query_info.compile_context, composition), "main")
                            .get()
                            ->get_program()
                            ->get_binary();
    return SpirvReflect(static_cast<const uint32_t*>(binary->getBufferPointer()),
                        binary->getBufferSize())
        .query_device_support(query_info);
}

} // namespace merian
