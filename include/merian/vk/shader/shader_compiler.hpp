#pragma once

#include "merian/vk/context.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace merian {

// A compiler for shaders.
//
// Include paths for the merian-nodes library and context extensions must be automatically added.
class ShaderCompiler;
using ShaderCompilerHandle = std::shared_ptr<ShaderCompiler>;
using WeakShaderCompilerHandle = std::weak_ptr<ShaderCompiler>;

class ShaderCompiler {
  public:
    class compilation_failed : public std::runtime_error {
      public:
        compilation_failed(const std::string& what) : std::runtime_error(what) {}
    };

  public:
    ShaderCompiler(const ContextHandle& context,
                   const std::vector<std::string>& user_include_paths = {},
                   const std::map<std::string, std::string>& user_macro_definitions = {});

    virtual ~ShaderCompiler() = default;

    // ------------------------------------------------

    void add_include_path(const std::string& include_path) {
        include_paths.emplace_back(include_path);
    }

    void add_macro_definition(const std::string& key, const std::string& value) {
        macro_definitions.emplace(key, value);
    }

    const std::vector<std::string>& get_include_paths() const {
        return include_paths;
    }

    const std::map<std::string, std::string>& get_macro_definitions() const {
        return macro_definitions;
    }

    virtual bool available() const = 0;

    void set_generate_debug_info(const bool enable) {
        generate_debug_info = enable;
    }

    bool generate_debug_info_enabled() const {
        return generate_debug_info;
    }

  private:
    std::vector<std::string> include_paths;
    std::map<std::string, std::string> macro_definitions;
    bool generate_debug_info;
};

} // namespace merian
