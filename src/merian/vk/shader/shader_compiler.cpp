#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/utils/vector.hpp"

#include "merian/vk/shader/shader_compiler_shaderc.hpp"
#include "merian/vk/shader/shader_compiler_system_glslangValidator.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"

namespace merian {

ShaderCompilerHandle
ShaderCompiler::get(const ContextHandle& context,
                    const std::vector<std::string>& user_include_paths,
                    const std::map<std::string, std::string>& user_macro_definitions) {
    ShaderCompilerHandle shaderc =
        std::make_shared<ShadercCompiler>(context, user_include_paths, user_macro_definitions);
    if (shaderc->available()) {
        SPDLOG_DEBUG("using shipped shaderc as default compiler");
        return shaderc;
    }

    ShaderCompilerHandle glslang_validator = std::make_shared<SystemGlslangValidatorCompiler>(
        context, user_include_paths, user_macro_definitions);
    if (glslang_validator->available()) {
        SPDLOG_DEBUG("using installed glslangValidator as default compiler");
        return glslang_validator;
    }

    ShaderCompilerHandle glslc =
        std::make_shared<SystemGlslcCompiler>(context, user_include_paths, user_macro_definitions);
    if (glslc->available()) {
        SPDLOG_DEBUG("using installed glslc as default compiler");
        return glslc;
    }

    return nullptr;
}

ShaderCompiler::ShaderCompiler(const ContextHandle& context,
                               const std::vector<std::string>& user_include_paths,
                               const std::map<std::string, std::string>& user_macro_definitions)
    : include_paths(user_include_paths), macro_definitions(user_macro_definitions) {

    insert_all(include_paths, context->get_default_shader_include_paths());
    macro_definitions.insert(context->get_default_shader_macro_definitions().begin(),
                             context->get_default_shader_macro_definitions().end());
}

ShaderCompiler::~ShaderCompiler(){};

} // namespace merian
