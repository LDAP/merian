#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/shader/entry_point.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_compiler.hpp"
#include "merian/shader/shader_module.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace merian {

// A compiler for shaders.
//
// Include paths for the merian-nodes library and context extensions must be automatically added.
class GLSLShaderCompiler;
using GLSLShaderCompilerHandle = std::shared_ptr<GLSLShaderCompiler>;
using WeakGLSLShaderCompilerHandle = std::weak_ptr<GLSLShaderCompiler>;

class GLSLShaderCompiler : public ShaderCompiler {
  public:
    inline static const std::map<std::string, vk::ShaderStageFlagBits> EXTENSION_SHADER_STAGE_MAP =
        {
            {".vert", vk::ShaderStageFlagBits::eVertex},
            {".tesc", vk::ShaderStageFlagBits::eTessellationControl},
            {".tese", vk::ShaderStageFlagBits::eTessellationEvaluation},
            {".geom", vk::ShaderStageFlagBits::eGeometry},
            {".frag", vk::ShaderStageFlagBits::eFragment},
            {".comp", vk::ShaderStageFlagBits::eCompute},
            {".mesh", vk::ShaderStageFlagBits::eMeshEXT},
            {".task", vk::ShaderStageFlagBits::eTaskEXT},
            {".rgen", vk::ShaderStageFlagBits::eRaygenKHR},
            {".rint", vk::ShaderStageFlagBits::eIntersectionKHR},
            {".rahit", vk::ShaderStageFlagBits::eAnyHitKHR},
            {".rchit", vk::ShaderStageFlagBits::eClosestHitKHR},
            {".rmiss", vk::ShaderStageFlagBits::eMissKHR},
            {".rcall", vk::ShaderStageFlagBits::eCallableKHR},
        };

    inline static const std::map<vk::ShaderStageFlagBits, std::string> SHADER_STAGE_EXTENSION_MAP =
        {
            {vk::ShaderStageFlagBits::eVertex, ".vert"},
            {vk::ShaderStageFlagBits::eTessellationControl, ".tesc"},
            {vk::ShaderStageFlagBits::eTessellationEvaluation, ".tese"},
            {vk::ShaderStageFlagBits::eGeometry, ".geom"},
            {vk::ShaderStageFlagBits::eFragment, ".frag"},
            {vk::ShaderStageFlagBits::eCompute, ".comp"},
            {vk::ShaderStageFlagBits::eMeshEXT, ".mesh"},
            {vk::ShaderStageFlagBits::eTaskEXT, ".task"},
            {vk::ShaderStageFlagBits::eRaygenKHR, ".rgen"},
            {vk::ShaderStageFlagBits::eIntersectionKHR, ".rint"},
            {vk::ShaderStageFlagBits::eAnyHitKHR, ".rahit"},
            {vk::ShaderStageFlagBits::eClosestHitKHR, ".rchit"},
            {vk::ShaderStageFlagBits::eMissKHR, ".rmiss"},
            {vk::ShaderStageFlagBits::eCallableKHR, ".rcall"},
        };

  public:
    GLSLShaderCompiler();

    virtual ~GLSLShaderCompiler() = default;

    // ------------------------------------------------

    // Attempt to guess the shader_kind from the file extension if shader_kind = std::nullopt.
    //
    // May throw compilation_failed.
    virtual std::vector<uint32_t> compile_glsl(
        const std::filesystem::path& path,
        const ShaderCompileContextHandle& shader_compile_context,
        const std::optional<vk::ShaderStageFlagBits> optional_shader_kind = std::nullopt) const {
        return compile_glsl(FileLoader::load_file(path), path.string(),
                            optional_shader_kind.value_or(guess_kind(path)),
                            shader_compile_context);
    }

    // May throw compilation_failed.
    virtual std::vector<uint32_t>
    compile_glsl(const std::string& source,
                 const std::string& source_name,
                 const vk::ShaderStageFlagBits shader_kind,
                 const ShaderCompileContextHandle& shader_compile_context) const = 0;

    // ------------------------------------------------

    ShaderModuleHandle compile_glsl_to_shadermodule(
        const ContextHandle& context,
        const std::filesystem::path& path,
        const ShaderCompileContextHandle& shader_compile_context,
        const std::optional<vk::ShaderStageFlagBits> optional_shader_kind = std::nullopt) const {
        const vk::ShaderStageFlagBits shader_kind = optional_shader_kind.value_or(guess_kind(path));
        std::vector<uint32_t> spv = compile_glsl(path, shader_compile_context, shader_kind);
        return ShaderModule::create(context, spv);
    }

    // uses the file_loader provided from context.
    ShaderModuleHandle find_compile_glsl_to_shadermodule(
        const ContextHandle& context,
        const std::filesystem::path& path,
        const ShaderCompileContextHandle& shader_compile_context,
        const std::optional<vk::ShaderStageFlagBits> optional_shader_kind = std::nullopt) const {

        const std::optional<std::filesystem::path> resolved = context->get_file_loader().find_file(path);
        if (!resolved) {
            throw compilation_failed{fmt::format("file {} not found", path.string())};
        }

        const vk::ShaderStageFlagBits shader_kind =
            optional_shader_kind.value_or(guess_kind(*resolved));
        std::vector<uint32_t> spv = compile_glsl(*resolved, shader_compile_context, shader_kind);
        return ShaderModule::create(context, spv);
    }

    // uses the file_loader provided from context.
    EntryPointHandle find_compile_glsl_to_entry_point(
        const ContextHandle& context,
        const std::filesystem::path& path,
        const ShaderCompileContextHandle& shader_compile_context,
        const std::string& entry_point_name = "main",
        const std::optional<vk::ShaderStageFlagBits> optional_shader_kind = std::nullopt) const {

        const std::optional<std::filesystem::path> resolved = context->get_file_loader().find_file(path);
        if (!resolved) {
            throw compilation_failed{fmt::format("file {} not found", path.string())};
        }

        const vk::ShaderStageFlagBits shader_kind =
            optional_shader_kind.value_or(guess_kind(*resolved));
        std::vector<uint32_t> spv = compile_glsl(*resolved, shader_compile_context, shader_kind);
        ShaderModuleHandle shader_module = ShaderModule::create(context, spv);

        return EntryPoint::create(entry_point_name, shader_kind, shader_module);
    }

    ShaderModuleHandle
    compile_glsl_to_shadermodule(const ContextHandle& context,
                                 const std::string& source,
                                 const std::string& source_name,
                                 const vk::ShaderStageFlagBits shader_kind,
                                 const ShaderCompileContextHandle& shader_compile_context) const {
        std::vector<uint32_t> spv =
            compile_glsl(source, source_name, shader_kind, shader_compile_context);
        return ShaderModule::create(context, spv);
    }

  private:
    static vk::ShaderStageFlagBits guess_kind(const std::filesystem::path& path) {
        std::string extension;
        if (path.extension().string() == ".glsl") {
            extension = std::filesystem::path(path.string().substr(0, path.string().size() - 5))
                            .extension()
                            .string();
        } else {
            extension = path.extension().string();
        }

        if (EXTENSION_SHADER_STAGE_MAP.contains(extension)) {
            return EXTENSION_SHADER_STAGE_MAP.at(extension);
        }

        throw compilation_failed{
            fmt::format("Shader kind could not be determined for path {}", path.string())};
    }
};

} // namespace merian
