#pragma once

#include "merian-graph/graph/node.hpp"

#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/utils/properties.hpp"

namespace merian {

enum class SVGFFilterType : uint8_t {
    ATROUS,
    BOX,
    SUBSAMPLED,
};

struct SVGFOptions {
    SVGFFilterType filter_type = SVGFFilterType::SUBSAMPLED;
    bool kaleidoscope = false;
    bool kaleidoscope_use_shmem = true;

    // True if an option changed.
    bool properties(Properties& config);
};

// The push constants of a filter level.
struct SVGFFilterParameters {
    // larger blurs more
    float depth = 10;
    // cos(alpha) of the lower threshold
    float normals = .8;
    // larger blurs more
    float luminance = 8;
    float z_bias_normals = -1.0;
    float z_bias_depth = -1.0;

    // True if a parameter changed.
    bool properties(Properties& config);
};

uint32_t svgf_filter_local_size(const ContextHandle& context, const SVGFOptions& options);

uint32_t svgf_vendor_local_size(const ContextHandle& context);

// The entry point "main" of an SVGF shader module for the gbuffer layout and irradiance format.
SlangProgramEntryPointHandle svgf_entry_point(const ShaderCompileContextHandle& compile_context,
                                              const SlangCompositionHandle& gbuffer_composition,
                                              const std::string& module_path,
                                              const SVGFOptions& options,
                                              const vk::Format irradiance_format);

DeviceSupportInfo svgf_device_support(const DeviceSupportQueryInfo& query_info,
                                      const std::string& module_path);

} // namespace merian
