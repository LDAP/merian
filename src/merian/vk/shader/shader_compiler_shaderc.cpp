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
            full_path = file_loader.find_file(requested_source, requesting_source);
            break;
        }
        default:
            throw ShaderCompiler::compilation_failed("unknown include type");
        }

        if (!full_path) {
            throw ShaderCompiler::compilation_failed(fmt::format(
                "Failed to find include: {} -> {}", requesting_source, requested_source));
        }

        ShadercIncludeInformation* incl_info = new ShadercIncludeInformation(*full_path);
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
            : full_path(full_path), content(FileLoader::load_file(full_path)) {
            include_result.source_name = this->full_path.c_str();
            include_result.source_name_length = this->full_path.size();
            include_result.content = content.c_str();
            include_result.content_length = content.size();
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

ShadercCompiler::ShadercCompiler(const std::vector<std::string>& include_paths,
                                 const std::map<std::string, std::string>& macro_definitions) {

    for (auto& [key, value] : macro_definitions) {
        compile_options.AddMacroDefinition(key, value);
    }

    auto includer = std::make_unique<FileIncluder>();
    for (auto& include_path : include_paths) {
        includer->get_file_loader().add_search_path(include_path);
    }
    compile_options.SetIncluder(std::move(includer));
}

ShadercCompiler::~ShadercCompiler() {}

std::vector<uint32_t> ShadercCompiler::compile_glsl(const std::string& source,
                                                    const std::string& source_name,
                                                    const vk::ShaderStageFlagBits shader_kind) {
    const shaderc_shader_kind kind = shaderc_shader_kind_for_stage_flag_bit(shader_kind);

    SPDLOG_DEBUG("preprocess {}", source_name);
    const auto preprocess_result =
        shader_compiler.PreprocessGlsl(source, kind, source_name.c_str(), compile_options);
    if (preprocess_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw ShaderCompiler::compilation_failed{preprocess_result.GetErrorMessage()};
    }

    SPDLOG_DEBUG("compile {}", source_name);
    const auto assembly_result = shader_compiler.CompileGlslToSpvAssembly(
        preprocess_result.begin(), preprocess_result.end() - preprocess_result.begin(), kind,
        source_name.c_str(), compile_options);
    if (assembly_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw ShaderCompiler::compilation_failed{assembly_result.GetErrorMessage()};
    }

    SPDLOG_DEBUG("assemble {}", source_name);
    const auto binary_result = shader_compiler.AssembleToSpv(
        assembly_result.begin(), assembly_result.end() - assembly_result.begin(), compile_options);
    if (binary_result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw ShaderCompiler::compilation_failed{binary_result.GetErrorMessage()};
    }

    return std::vector<uint32_t>(binary_result.begin(), binary_result.end());
}

} // namespace merian
