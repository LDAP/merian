#include "merian/shader/glsl_shader_compiler_glslang.hpp"

#ifdef MERIAN_GLSLANG_ENABLED
#include "glslang/Public/ResourceLimits.h"
#include "glslang/Public/ShaderLang.h"
#include "glslang/SPIRV/GlslangToSpv.h"
#endif

#include <fmt/format.h>

namespace merian {

#ifdef MERIAN_GLSLANG_ENABLED

namespace {

EShLanguage vk_stage_to_esh_language(const vk::ShaderStageFlagBits shader_kind) {
    switch (shader_kind) {
    case vk::ShaderStageFlagBits::eVertex:
        return EShLangVertex;
    case vk::ShaderStageFlagBits::eTessellationControl:
        return EShLangTessControl;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        return EShLangTessEvaluation;
    case vk::ShaderStageFlagBits::eGeometry:
        return EShLangGeometry;
    case vk::ShaderStageFlagBits::eFragment:
        return EShLangFragment;
    case vk::ShaderStageFlagBits::eCompute:
        return EShLangCompute;
    case vk::ShaderStageFlagBits::eRaygenKHR:
        return EShLangRayGen;
    case vk::ShaderStageFlagBits::eIntersectionKHR:
        return EShLangIntersect;
    case vk::ShaderStageFlagBits::eAnyHitKHR:
        return EShLangAnyHit;
    case vk::ShaderStageFlagBits::eClosestHitKHR:
        return EShLangClosestHit;
    case vk::ShaderStageFlagBits::eMissKHR:
        return EShLangMiss;
    case vk::ShaderStageFlagBits::eCallableKHR:
        return EShLangCallable;
    case vk::ShaderStageFlagBits::eTaskEXT:
        return EShLangTask;
    case vk::ShaderStageFlagBits::eMeshEXT:
        return EShLangMesh;
    default:
        throw GLSLShaderCompiler::compilation_failed{
            fmt::format("shader kind {} unsupported.", vk::to_string(shader_kind))};
    }
}

glslang::EShTargetClientVersion vk_api_version_to_client_version(const uint32_t vk_api_version) {
    if (vk_api_version >= VK_API_VERSION_1_4)
        return glslang::EShTargetVulkan_1_4;
    if (vk_api_version >= VK_API_VERSION_1_3)
        return glslang::EShTargetVulkan_1_3;
    if (vk_api_version >= VK_API_VERSION_1_2)
        return glslang::EShTargetVulkan_1_2;
    if (vk_api_version >= VK_API_VERSION_1_1)
        return glslang::EShTargetVulkan_1_1;
    return glslang::EShTargetVulkan_1_0;
}

glslang::EShTargetLanguageVersion spirv_version_to_target_version(const SpirvVersion spirv_ver) {
    if (spirv_ver >= MERIAN_SPIRV_VERSION_1_6)
        return glslang::EShTargetSpv_1_6;
    if (spirv_ver >= MERIAN_SPIRV_VERSION_1_5)
        return glslang::EShTargetSpv_1_5;
    if (spirv_ver >= MERIAN_SPIRV_VERSION_1_4)
        return glslang::EShTargetSpv_1_4;
    if (spirv_ver >= MERIAN_SPIRV_VERSION_1_3)
        return glslang::EShTargetSpv_1_3;
    if (spirv_ver >= MERIAN_SPIRV_VERSION_1_2)
        return glslang::EShTargetSpv_1_2;
    if (spirv_ver >= MERIAN_SPIRV_VERSION_1_1)
        return glslang::EShTargetSpv_1_1;
    return glslang::EShTargetSpv_1_0;
}

class MerianGlslangIncluder : public glslang::TShader::Includer {
  public:
    MerianGlslangIncluder(const FileLoader& file_loader) : file_loader(file_loader) {}

    IncludeResult* includeSystem(const char* header_name,
                                 const char* includer_name,
                                 size_t /*inclusion_depth*/) override {
        SPDLOG_TRACE("requested system include {} -> {}", includer_name, header_name);
        const auto resolved = file_loader.find_file(header_name);
        if (!resolved)
            SPDLOG_WARN("Failed to find system include: {} -> {}", includer_name, header_name);
        return make_result(resolved);
    }

    IncludeResult* includeLocal(const char* header_name,
                                const char* includer_name,
                                size_t /*inclusion_depth*/) override {
        SPDLOG_TRACE("requested local include {} -> {}", includer_name, header_name);
        const auto resolved =
            file_loader.find_file(header_name, std::filesystem::path(includer_name));
        if (!resolved)
            SPDLOG_WARN("Failed to find local include: {} -> {}", includer_name, header_name);
        return make_result(resolved);
    }

    void releaseInclude(IncludeResult* result) override {
        if (result) {
            delete static_cast<std::string*>(result->userData);
            delete result;
        }
    }

  private:
    IncludeResult* make_result(const std::optional<std::filesystem::path>& resolved) {
        if (!resolved) {
            // Empty headerName signals failure to glslang; headerData becomes the error message.
            auto* msg = new std::string("file not found");
            return new IncludeResult("", msg->c_str(), msg->size(), msg);
        }
        auto* content = new std::string(FileLoader::load_file_as_string(*resolved));
        return new IncludeResult(resolved->string(), content->c_str(), content->size(), content);
    }

    const FileLoader& file_loader;
};

} // namespace

GlslangCompiler::GlslangCompiler() : GLSLShaderCompiler() {
    glslang::InitializeProcess();
}

GlslangCompiler::~GlslangCompiler() {
    glslang::FinalizeProcess();
}

BlobHandle
GlslangCompiler::compile_glsl(const std::string& source,
                              const std::string& source_name,
                              const vk::ShaderStageFlagBits shader_kind,
                              const ShaderCompileContextHandle& shader_compile_context) const {
    const EShLanguage stage = vk_stage_to_esh_language(shader_kind);

    glslang::TShader shader(stage);

    const char* source_cstr = source.c_str();
    const char* source_name_cstr = source_name.c_str();
    shader.setStringsWithLengthsAndNames(&source_cstr, nullptr, &source_name_cstr, 1);

    // Inject preprocessor macros as a preamble
    std::string preamble;
    for (const auto& [key, value] : shader_compile_context->get_preprocessor_macros()) {
        if (value.empty()) {
            preamble += fmt::format("#define {}\n", key);
        } else {
            preamble += fmt::format("#define {} {}\n", key, value);
        }
    }
    if (!preamble.empty()) {
        shader.setPreamble(preamble.c_str());
    }

    const uint32_t target_vk_version = shader_compile_context->get_target_vk_api_version();
    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan,
                        vk_api_version_to_client_version(target_vk_version));
    shader.setEnvTarget(glslang::EShTargetSpv,
                        spirv_version_to_target_version(shader_compile_context->get_target()));

    if (shader_compile_context->should_generate_debug_info()) {
        shader.setDebugInfo(true);
    }
    shader.setEnhancedMsgs();

    const EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);

    MerianGlslangIncluder includer(shader_compile_context->get_search_path_file_loader());

    SPDLOG_DEBUG("compile {}", source_name);
    if (!shader.parse(GetDefaultResources(), 100, false, messages, includer)) {
        throw compilation_failed{fmt::format("glslang compilation failed for {}:\n{}", source_name,
                                             shader.getInfoLog())};
    }

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(messages)) {
        throw compilation_failed{
            fmt::format("glslang link failed for {}:\n{}", source_name, program.getInfoLog())};
    }

    if (!program.mapIO()) {
        throw compilation_failed{
            fmt::format("glslang mapIO failed for {}:\n{}", source_name, program.getInfoLog())};
    }

    glslang::SpvOptions spv_options;
    spv_options.generateDebugInfo = shader_compile_context->should_generate_debug_info();
    spv_options.emitNonSemanticShaderDebugInfo =
        shader_compile_context->should_generate_debug_info();
    spv_options.emitNonSemanticShaderDebugSource =
        shader_compile_context->should_generate_debug_info();
    spv_options.disableOptimizer = shader_compile_context->get_optimization_level() == 0 ||
                                   shader_compile_context->should_generate_debug_info();

    spv::SpvBuildLogger logger;
    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &logger, &spv_options);

    const std::string logger_messages = logger.getAllMessages();
    if (!logger_messages.empty()) {
        SPDLOG_WARN("glslang SPIRV messages for {}: {}", source_name, logger_messages);
    }

    return std::make_shared<VectorBlob<uint32_t>>(std::move(spirv));
}

bool GlslangCompiler::available() const {
    return true;
}
#else

GlslangCompiler::GlslangCompiler() : GLSLShaderCompiler() {}

GlslangCompiler::~GlslangCompiler() {}

BlobHandle GlslangCompiler::compile_glsl(
    [[maybe_unused]] const std::string& source,
    [[maybe_unused]] const std::string& source_name,
    [[maybe_unused]] const vk::ShaderStageFlagBits shader_kind,
    [[maybe_unused]] const ShaderCompileContextHandle& shader_compile_context) const {
    throw merian::ShaderCompiler::compilation_failed{
        "glslang is not available (was not found or enabled at compile time)"};
}

bool GlslangCompiler::available() const {
    return false;
}

#endif

} // namespace merian
