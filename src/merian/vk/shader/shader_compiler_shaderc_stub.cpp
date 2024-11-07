#include "merian/vk/shader/shader_compiler_shaderc.hpp"

#include <map>

namespace merian {

ShadercCompiler::ShadercCompiler(const ContextHandle& context,
                                 const std::vector<std::string>& include_paths,
                                 const std::map<std::string, std::string>& macro_definitions)
    : ShaderCompiler(context, include_paths, macro_definitions) {}

ShadercCompiler::~ShadercCompiler() {}

std::vector<uint32_t>
ShadercCompiler::compile_glsl([[maybe_unused]] const std::string& source,
                              [[maybe_unused]] const std::string& source_name,
                              [[maybe_unused]] const vk::ShaderStageFlagBits shader_kind) {
    throw merian::ShaderCompiler::compilation_failed{
        "shaderc is not available (was not found or enabled at compile time)"};
}

bool ShadercCompiler::available() const {
    return false;
}

} // namespace merian
