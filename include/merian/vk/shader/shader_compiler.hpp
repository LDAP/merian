#pragma once

#include <memory>
#include <stdexcept>
#include <string>

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
    ShaderCompiler();

    virtual ~ShaderCompiler() = default;

    // ------------------------------------------------

    virtual bool available() const = 0;

  private:
};

} // namespace merian
