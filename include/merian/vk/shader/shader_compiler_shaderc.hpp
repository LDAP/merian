#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include <map>

#ifdef __has_include
#if !__has_include(<shaderc/shaderc.hpp>)
static_assert(false, "shaderc is required for ShadercCompiler");
#else
#include <shaderc/shaderc.hpp>
#endif
#else
#include <shaderc/shaderc.hpp>
#endif

namespace merian {

class ShadercCompiler : public ShaderCompiler {
  public:
    ShadercCompiler(const std::vector<std::string>& include_paths = {},
                    const std::map<std::string, std::string>& macro_definitions = {});

    ~ShadercCompiler();

    std::vector<uint32_t> compile_glsl(const std::string& source,
                                       const std::string& source_name,
                                       const vk::ShaderStageFlagBits shader_kind) override;

  private:
    shaderc::Compiler shader_compiler;
    shaderc::CompileOptions compile_options;
};

} // namespace merian
