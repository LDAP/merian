#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/utils/vector.hpp"

namespace merian {

ShaderCompiler::ShaderCompiler(const ContextHandle& context,
                               const std::vector<std::string>& user_include_paths,
                               const std::map<std::string, std::string>& user_macro_definitions)
    : include_paths(user_include_paths.begin(), user_include_paths.end()),
      macro_definitions(user_macro_definitions) {

    insert_all(include_paths, context->get_default_shader_include_paths());
    macro_definitions.insert(context->get_default_shader_macro_definitions().begin(),
                             context->get_default_shader_macro_definitions().end());
    generate_debug_info = Context::IS_DEBUG_BUILD;
}

} // namespace merian
