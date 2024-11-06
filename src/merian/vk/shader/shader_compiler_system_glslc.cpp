#include "merian/vk/shader/shader_compiler_system_glslc.hpp"

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
    : ShaderCompiler(context, user_include_paths, user_macro_definitions), context(context) {}

SystemGlslcCompiler::~SystemGlslcCompiler() {}



std::vector<uint32_t>
SystemGlslcCompiler::compile_glsl(const std::string& source,
                                  [[maybe_unused]] const std::string& source_name,
                                  const vk::ShaderStageFlagBits shader_kind) {
    std::vector<std::string> command = {subprocess::find_program("glslc")};

    if (context->vk_api_version == VK_API_VERSION_1_0) {
        command.emplace_back("--target-env=vulkan1.0");
    } else if (context->vk_api_version == VK_API_VERSION_1_1) {
        command.emplace_back("--target-env=vulkan1.1");
    } else if (context->vk_api_version == VK_API_VERSION_1_2) {
        command.emplace_back("--target-env=vulkan1.2");
    } else {
        command.emplace_back("--target-env=vulkan1.3");
    }

    if (!SHADER_STAGE_EXTENSION_MAP.contains(shader_kind)) {
        throw compilation_failed{
            fmt::format("shader kind {} unsupported.", vk::to_string(shader_kind))};
    }

    command.emplace_back(fmt::format("-fshader-stage={}", SHADER_STAGE_EXTENSION_MAP.at(shader_kind).substr(1)));

    for (const auto& inc_dir : get_include_paths()) {
        command.emplace_back("-I");
        command.emplace_back(inc_dir);
    }
    for (const auto& macro_def : get_macro_definitions()) {
        command.emplace_back(fmt::format("-D{}={}", macro_def.first, macro_def.second));
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
        throw compilation_failed{fmt::format("glslc command failed:\n{}\n\n{}\n\n{}",
                                             process.cout, process.cerr, fmt::join(command, " "))};
    }

    std::vector<uint32_t> spv((process.cout.size() + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    memcpy(spv.data(), process.cout.data(), process.cout.size());

    return spv;
}

bool SystemGlslcCompiler::available() const {
    return !subprocess::find_program("glslc").empty();
}

} // namespace merian
