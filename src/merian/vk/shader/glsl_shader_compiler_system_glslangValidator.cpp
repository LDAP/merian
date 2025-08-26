#include "merian/vk/shader/glsl_shader_compiler_system_glslangValidator.hpp"
#include "merian/utils/filesystem.hpp"

#include "subprocess/ProcessBuilder.hpp"
#include "subprocess/basic_types.hpp"
#include "subprocess/shell_utils.hpp"
#include <fmt/ranges.h>

namespace merian {

// Include paths for the merian-nodes library are automatically added
SystemGlslangValidatorCompiler::SystemGlslangValidatorCompiler()
    : GLSLShaderCompiler(), compiler_executable(subprocess::find_program("glslangValidator")) {}

SystemGlslangValidatorCompiler::~SystemGlslangValidatorCompiler() {}

std::vector<uint32_t> SystemGlslangValidatorCompiler::compile_glsl(
    const std::string& source,
    const std::string& source_name,
    const vk::ShaderStageFlagBits shader_kind,
    const CompilationSessionDescription& compilation_session_description) const {
    if (compiler_executable.empty()) {
        throw compilation_failed{"compiler not available"};
    }

    std::vector<std::string> command = {compiler_executable};

    command.emplace_back("--target-env");
    if (compilation_session_description.get_target_vk_api_version() == VK_API_VERSION_1_0) {
        command.emplace_back("vulkan1.0");
    } else if (compilation_session_description.get_target_vk_api_version() == VK_API_VERSION_1_1) {
        command.emplace_back("vulkan1.1");
    } else if (compilation_session_description.get_target_vk_api_version() == VK_API_VERSION_1_2) {
        command.emplace_back("vulkan1.2");
    } else {
        command.emplace_back("vulkan1.3");
    }

    command.emplace_back("--stdin");

    if (!SHADER_STAGE_EXTENSION_MAP.contains(shader_kind)) {
        throw compilation_failed{
            fmt::format("shader kind {} unsupported.", vk::to_string(shader_kind))};
    }

    command.emplace_back("-S");
    command.emplace_back(SHADER_STAGE_EXTENSION_MAP.at(shader_kind).substr(1));

    const std::filesystem::path source_path(source_name);
    if (FileLoader::exists(source_path)) {
        const std::filesystem::path parent_path = source_path.parent_path();
        command.emplace_back(fmt::format("-I{}", parent_path.string()));
    }
    for (const auto& inc_dir : compilation_session_description.get_include_paths()) {
        command.emplace_back(fmt::format("-I{}", inc_dir));
    }
    for (const auto& [key, value] : compilation_session_description.get_preprocessor_defines()) {
        command.emplace_back(fmt::format("-D{}={}", key, value));
    }

    if (compilation_session_description.should_generate_debug_info()) {
        command.emplace_back("-g");
    }

    const std::string output_file = temporary_file();
    command.emplace_back("-o");
    command.emplace_back(output_file);

    SPDLOG_DEBUG("running command {}", fmt::join(command, " "));
    subprocess::CompletedProcess process =
        subprocess::run(command, subprocess::RunBuilder()
                                     .cin(source)
                                     .cerr(subprocess::PipeOption::pipe)
                                     .cout(subprocess::PipeOption::pipe));

    if (process.returncode != 0) {
        throw compilation_failed{
            fmt::format("glslangValidator command failed compiling {}:\n{}\n\n{}\n\n{}",
                        source_name, process.cout, process.cerr, fmt::join(command, " "))};
    }

    std::vector<uint32_t> spv = FileLoader::load_file<uint32_t>(output_file);

    std::filesystem::remove(output_file);

    return spv;
}

bool SystemGlslangValidatorCompiler::available() const {
    return !compiler_executable.empty();
}

} // namespace merian
