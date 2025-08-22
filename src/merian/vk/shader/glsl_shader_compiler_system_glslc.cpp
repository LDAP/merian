#include "merian/vk/shader/glsl_shader_compiler_system_glslc.hpp"

#include "subprocess/ProcessBuilder.hpp"
#include "subprocess/basic_types.hpp"
#include "subprocess/shell_utils.hpp"

#include <fmt/ranges.h>

namespace merian {

// Include paths for the merian-nodes library are automatically added
SystemGlslcCompiler::SystemGlslcCompiler(
    const ContextHandle& context,
    const std::vector<std::string>& user_include_paths,
    const std::map<std::string, std::string>& user_macro_definitions)
    : GLSLShaderCompiler(context, user_include_paths, user_macro_definitions),
      compiler_executable(subprocess::find_program("glslc")) {
    if (context->vk_api_version == VK_API_VERSION_1_0) {
        target_env_arg = "--target-env=vulkan1.0";
    } else if (context->vk_api_version == VK_API_VERSION_1_1) {
        target_env_arg = "--target-env=vulkan1.1";
    } else if (context->vk_api_version == VK_API_VERSION_1_2) {
        target_env_arg = "--target-env=vulkan1.2";
    } else {
        target_env_arg = "--target-env=vulkan1.3";
    }
}

SystemGlslcCompiler::~SystemGlslcCompiler() {}

std::vector<uint32_t> SystemGlslcCompiler::compile_glsl(
    const std::string& source,
    const std::string& source_name,
    const vk::ShaderStageFlagBits shader_kind,
    const std::vector<std::string>& additional_include_paths,
    const std::map<std::string, std::string>& additional_macro_definitions) const {
    if (compiler_executable.empty()) {
        throw compilation_failed{"compiler not available"};
    }

    std::vector<std::string> command = {compiler_executable};

    command.emplace_back(target_env_arg);

    if (!SHADER_STAGE_EXTENSION_MAP.contains(shader_kind)) {
        throw compilation_failed{
            fmt::format("shader kind {} unsupported.", vk::to_string(shader_kind))};
    }

    command.emplace_back(
        fmt::format("-fshader-stage={}", SHADER_STAGE_EXTENSION_MAP.at(shader_kind).substr(1)));

    const std::filesystem::path source_path(source_name);
    if (FileLoader::exists(source_path)) {
        const std::filesystem::path parent_path = source_path.parent_path();
        command.emplace_back("-I");
        command.emplace_back(parent_path.string());
    }
    for (const auto& inc_dir : get_include_paths()) {
        command.emplace_back("-I");
        command.emplace_back(inc_dir);
    }
    for (const auto& inc_dir : additional_include_paths) {
        command.emplace_back("-I");
        command.emplace_back(inc_dir);
    }
    for (const auto& [key, value] : get_macro_definitions()) {
        command.emplace_back(fmt::format("-D{}={}", key, value));
    }
    for (const auto& [key, value] : additional_macro_definitions) {
        command.emplace_back(fmt::format("-D{}={}", key, value));
    }

    if (generate_debug_info_enabled()) {
        command.emplace_back("-g");
    }

    // turn on optimization
    command.emplace_back("-O");

    command.emplace_back("-");

    command.emplace_back("-o");
    command.emplace_back("-");

    SPDLOG_DEBUG("running command {}", fmt::join(command, " "));
    subprocess::CompletedProcess process =
        subprocess::run(command, subprocess::RunBuilder()
                                     .cin(source)
                                     .cerr(subprocess::PipeOption::pipe)
                                     .cout(subprocess::PipeOption::pipe));

    if (process.returncode != 0) {
        throw compilation_failed{fmt::format("glslc command failed compiling {}:\n{}\n\n{}\n\n{}",
                                             source_name, process.cout, process.cerr,
                                             fmt::join(command, " "))};
    }

    std::vector<uint32_t> spv((process.cout.size() + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    memcpy(spv.data(), process.cout.data(), process.cout.size());

    return spv;
}

bool SystemGlslcCompiler::available() const {
    return !compiler_executable.empty();
}

} // namespace merian
