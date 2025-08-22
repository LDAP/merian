#pragma once

#include "merian/vk/shader/glsl_shader_compiler.hpp"
#include <map>

#ifdef MERIAN_SHADERC_AVAILABLE
#include <shaderc/shaderc.hpp>
#endif

namespace merian {

class ShadercCompiler : public GLSLShaderCompiler {
  public:
    ShadercCompiler(const ContextHandle& context,
                    const std::vector<std::string>& include_paths = {},
                    const std::map<std::string, std::string>& macro_definitions = {});

    ~ShadercCompiler();

    std::vector<uint32_t> compile_glsl(
        const std::string& source,
        const std::string& source_name,
        const vk::ShaderStageFlagBits shader_kind,
        const std::vector<std::string>& additional_include_paths = {},
        const std::map<std::string, std::string>& additional_macro_definitions = {}) const override;

    bool available() const override;

  private:
    const uint32_t vk_api_version;
#ifdef MERIAN_SHADERC_AVAILABLE
    shaderc::Compiler shader_compiler{};
#endif
};

} // namespace merian
