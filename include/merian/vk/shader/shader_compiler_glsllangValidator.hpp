#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include <map>

namespace merian {

// Uses glslangValidator executable to compile shaders.
class GLSLLangValidatorCompiler : public ShaderCompiler {
  public:
    // Include paths for the merian-nodes library are automatically added
    GLSLLangValidatorCompiler(const std::vector<std::string>& include_paths = {},
                              const std::map<std::string, std::string>& macro_definitions = {});

    ~GLSLLangValidatorCompiler();

    std::vector<uint32_t> compile_glsl(const std::string& source,
                                       const std::string& source_name,
                                       const vk::ShaderStageFlagBits shader_kind) override;
};

} // namespace merian
