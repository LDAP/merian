#include "merian/vk/shader/shader_compiler_shaderc.hpp"

#include <map>

namespace merian {

class FileIncluder final : public shaderc::CompileOptions::IncluderInterface {
  public:
    FileIncluder() = default;

    shaderc_include_result* GetInclude(const char* requested_source,
                                       shaderc_include_type type,
                                       const char* requesting_source,
                                       [[maybe_unused]] size_t include_depth) override {
        SPDLOG_TRACE("requested include {} -> {}", requesting_source, requested_source);

        std::optional<std::filesystem::path> full_path;

        switch (type) {
        case shaderc_include_type_standard: {
            full_path = file_loader.find_file(requested_source);
            break;
        }
        case shaderc_include_type_relative: {
            full_path =
                file_loader.find_file(requested_source, std::filesystem::path(requesting_source));
            break;
        }
        default: { // do nothing to trigger error.
        }
        }

        ShadercIncludeInformation* incl_info;

        if (full_path) {
            incl_info = new ShadercIncludeInformation(*full_path);
        } else {
            SPDLOG_WARN("Failed to find include: {} -> {}", requesting_source, requested_source);
            incl_info = new ShadercIncludeInformation();
        }

        incl_info->include_result.user_data = incl_info;
        return &incl_info->include_result;
    }

    void ReleaseInclude(shaderc_include_result* data) override {
        ShadercIncludeInformation* incl_info =
            static_cast<ShadercIncludeInformation*>(data->user_data);
        delete incl_info;
    }

    // the file loader to resolve includes
    FileLoader& get_file_loader() {
        return file_loader;
    }

  private:
    struct ShadercIncludeInformation {

        ShadercIncludeInformation(const std::filesystem::path& full_path)
            : full_path(full_path.string()), content(FileLoader::load_file(full_path)) {
            include_result.source_name = this->full_path.c_str();
            include_result.source_name_length = this->full_path.size();
            include_result.content = content.c_str();
            include_result.content_length = content.size();
        }

        // failed
        ShadercIncludeInformation() {
            include_result.source_name = "";
            include_result.source_name_length = 0;
            include_result.content = "";
            include_result.content_length = 0;
        }

        const std::string full_path;
        const std::string content;
        shaderc_include_result include_result;
    };

    FileLoader file_loader;
};

static shaderc_shader_kind
shaderc_shader_kind_for_stage_flag_bit(const vk::ShaderStageFlagBits shader_kind) {
    switch (shader_kind) {
    case vk::ShaderStageFlagBits::eVertex:
        return shaderc_shader_kind::shaderc_vertex_shader;
    case vk::ShaderStageFlagBits::eTessellationControl:
        return shaderc_shader_kind::shaderc_tess_control_shader;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        return shaderc_shader_kind::shaderc_tess_evaluation_shader;
    case vk::ShaderStageFlagBits::eGeometry:
        return shaderc_shader_kind::shaderc_geometry_shader;
    case vk::ShaderStageFlagBits::eFragment:
        return shaderc_shader_kind::shaderc_fragment_shader;
    case vk::ShaderStageFlagBits::eCompute:
        return shaderc_shader_kind::shaderc_compute_shader;
    case vk::ShaderStageFlagBits::eAnyHitKHR:
        return shaderc_shader_kind::shaderc_anyhit_shader;
    case vk::ShaderStageFlagBits::eCallableKHR:
        return shaderc_shader_kind::shaderc_callable_shader;
    case vk::ShaderStageFlagBits::eClosestHitKHR:
        return shaderc_shader_kind::shaderc_closesthit_shader;
    case vk::ShaderStageFlagBits::eMeshEXT:
        return shaderc_shader_kind::shaderc_mesh_shader;
    case vk::ShaderStageFlagBits::eMissKHR:
        return shaderc_shader_kind::shaderc_miss_shader;
    case vk::ShaderStageFlagBits::eRaygenKHR:
        return shaderc_shader_kind::shaderc_raygen_shader;
    case vk::ShaderStageFlagBits::eIntersectionKHR:
        return shaderc_shader_kind::shaderc_intersection_shader;
    default:
        throw ShaderCompiler::compilation_failed("shader kind not supported");
    }
}

ShadercCompiler::ShadercCompiler(const ContextHandle& context,
                                 const std::vector<std::string>& user_include_paths,
                                 const std::map<std::string, std::string>& user_macro_definitions)
    : ShaderCompiler(context, user_include_paths, user_macro_definitions),
      vk_api_version(context->vk_api_version) {}

ShadercCompiler::~ShadercCompiler() {}

std::vector<uint32_t> ShadercCompiler::compile_glsl(
    const std::string& source,
    const std::string& source_name,
    const vk::ShaderStageFlagBits shader_kind,
    const std::vector<std::string>& additional_include_paths,
    const std::map<std::string, std::string>& additional_macro_definitions) const {
    const shaderc_shader_kind kind = shaderc_shader_kind_for_stage_flag_bit(shader_kind);

    shaderc::CompileOptions compile_options;
    if (generate_debug_info_enabled())
        compile_options.SetGenerateDebugInfo();

    for (const auto& [key, value] : get_macro_definitions()) {
        compile_options.AddMacroDefinition(key, value);
    }
    for (const auto& [key, value] : additional_macro_definitions) {
        compile_options.AddMacroDefinition(key, value);
    }

    auto includer = std::make_unique<FileIncluder>();
    for (const auto& include_path : get_include_paths()) {
        includer->get_file_loader().add_search_path(include_path);
    }
    for (const auto& include_path : additional_include_paths) {
        includer->get_file_loader().add_search_path(include_path);
    }
    compile_options.SetIncluder(std::move(includer));
    compile_options.SetOptimizationLevel(
        shaderc_optimization_level::shaderc_optimization_level_performance);

    if (vk_api_version == VK_API_VERSION_1_0) {
        compile_options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                             shaderc_env_version_vulkan_1_0);
    } else if (vk_api_version == VK_API_VERSION_1_1) {
        compile_options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                             shaderc_env_version_vulkan_1_1);
    } else if (vk_api_version == VK_API_VERSION_1_2) {
        compile_options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                             shaderc_env_version_vulkan_1_2);
    } else {
        compile_options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                             shaderc_env_version_vulkan_1_3);
    }

    SPDLOG_DEBUG("preprocess {}", source_name);
    const auto preprocess_result =
        shader_compiler.PreprocessGlsl(source, kind, source_name.c_str(), compile_options);
    if (preprocess_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw ShaderCompiler::compilation_failed{preprocess_result.GetErrorMessage()};
    }

    SPDLOG_DEBUG("compile and assemble {}", source_name);
    const auto binary_result = shader_compiler.CompileGlslToSpv(
        preprocess_result.begin(), preprocess_result.end() - preprocess_result.begin(), kind,
        source_name.data(), compile_options);
    if (binary_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw ShaderCompiler::compilation_failed{binary_result.GetErrorMessage()};
    }

    return std::vector<uint32_t>(binary_result.begin(), binary_result.end());
}

bool ShadercCompiler::available() const {
    return true;
}

} // namespace merian
