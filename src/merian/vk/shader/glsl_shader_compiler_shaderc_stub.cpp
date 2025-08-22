#include "merian/vk/shader/glsl_shader_compiler_shaderc.hpp"

#include <map>

namespace merian {

ShadercCompiler::ShadercCompiler(const ContextHandle& context,
                                 const std::vector<std::string>& include_paths,
                                 const std::map<std::string, std::string>& macro_definitions)
    : ShaderCompiler(context, include_paths, macro_definitions), vk_api_version(-1) {}

ShadercCompiler::~ShadercCompiler() {}

std::vector<uint32_t> ShadercCompiler::compile_glsl(
    [[maybe_unused]] const std::string& source,
    [[maybe_unused]] const std::string& source_name,
    [[maybe_unused]] const vk::ShaderStageFlagBits shader_kind,
    [[maybe_unused]] const std::vector<std::string>& additional_include_paths,
    [[maybe_unused]] const std::map<std::string, std::string>& additional_macro_definitions) const {
    throw merian::ShaderCompiler::compilation_failed{
        "shaderc is not available (was not found or enabled at compile time)"};
}

bool ShadercCompiler::available() const {
    return false;
}

} // namespace merian
