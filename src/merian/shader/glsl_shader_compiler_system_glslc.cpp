#include "merian/shader/glsl_shader_compiler_system_glslc.hpp"

#include "subprocess/ProcessBuilder.hpp"
#include "subprocess/basic_types.hpp"
#include "subprocess/shell_utils.hpp"

#include <fmt/ranges.h>

namespace merian {

// Include paths for the merian-nodes library are automatically added
SystemGlslcCompiler::SystemGlslcCompiler()
    : GLSLShaderCompiler(), compiler_executable(subprocess::find_program("glslc")) {}

SystemGlslcCompiler::~SystemGlslcCompiler() {}

std::vector<uint32_t>
SystemGlslcCompiler::compile_glsl(const std::string& source,
                                  const std::string& source_name,
                                  const vk::ShaderStageFlagBits shader_kind,
                                  const ShaderCompileContextHandle& shader_compile_context) const {
    if (compiler_executable.empty()) {
        throw compilation_failed{"compiler not available"};
    }

    std::vector<std::string> command = {compiler_executable};

    if (shader_compile_context->get_target_vk_api_version() == VK_API_VERSION_1_0) {
        command.emplace_back("--target-env=vulkan1.0");
    } else if (shader_compile_context->get_target_vk_api_version() == VK_API_VERSION_1_1) {
        command.emplace_back("--target-env=vulkan1.1");
    } else if (shader_compile_context->get_target_vk_api_version() == VK_API_VERSION_1_2) {
        command.emplace_back("--target-env=vulkan1.2");
    } else {
        command.emplace_back("--target-env=vulkan1.3");
    };

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
    for (const auto& inc_dir : shader_compile_context->get_search_path_file_loader()) {
        command.emplace_back("-I");
        command.emplace_back(inc_dir.string());
    }
    for (const auto& [key, value] : shader_compile_context->get_preprocessor_macros()) {
        command.emplace_back(fmt::format("-D{}={}", key, value));
    }

    if (shader_compile_context->should_generate_debug_info()) {
        command.emplace_back("-g");
    }

    // turn on optimization
    if (shader_compile_context->get_optimization_level() > 0) {
        command.emplace_back("-O");
    }

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
