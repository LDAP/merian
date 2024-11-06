#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include <map>

#ifdef MERIAN_SHADERC_FOUND
#include <shaderc/shaderc.hpp>
#endif

namespace merian {

class ShadercCompiler : public ShaderCompiler {
  public:
    ShadercCompiler(const ContextHandle& context,
                    const std::vector<std::string>& include_paths = {},
                    const std::map<std::string, std::string>& macro_definitions = {});

    ~ShadercCompiler();

    std::vector<uint32_t> compile_glsl(const std::string& source,
                                       const std::string& source_name,
                                       const vk::ShaderStageFlagBits shader_kind) override;

    bool available() const override;

  private:
#ifdef MERIAN_SHADERC_FOUND
    shaderc::Compiler shader_compiler;
    shaderc::CompileOptions compile_options;
#endif
};

} // namespace merian
