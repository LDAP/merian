#include "merian/vk/shader/shader_compiler_system_glslangValidator.hpp"
#include "merian/utils/filesystem.hpp"

#include "subprocess/ProcessBuilder.hpp"
#include "subprocess/basic_types.hpp"
#include "subprocess/shell_utils.hpp"
#include <fmt/ranges.h>

namespace merian {

// Include paths for the merian-nodes library are automatically added
SystemGlslangValidatorCompiler::SystemGlslangValidatorCompiler(
    const ContextHandle& context,
    const std::vector<std::string>& user_include_paths,
    const std::map<std::string, std::string>& user_macro_definitions)
    : ShaderCompiler(context, user_include_paths, user_macro_definitions), context(context) {}

SystemGlslangValidatorCompiler::~SystemGlslangValidatorCompiler() {}

std::vector<uint32_t>
SystemGlslangValidatorCompiler::compile_glsl(const std::string& source,
                                             [[maybe_unused]] const std::string& source_name,
                                             const vk::ShaderStageFlagBits shader_kind) {
    std::vector<std::string> command = {subprocess::find_program("glslangValidator")};

    command.emplace_back("--target-env");
    if (context->vk_api_version == VK_API_VERSION_1_0) {
        command.emplace_back("vulkan1.0");
    } else if (context->vk_api_version == VK_API_VERSION_1_1) {
        command.emplace_back("vulkan1.1");
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

    for (const auto& inc_dir : get_include_paths()) {
        command.emplace_back(fmt::format("-I{}", inc_dir));
    }
    for (const auto& macro_def : get_macro_definitions()) {
        command.emplace_back(fmt::format("-D{}={}", macro_def.first, macro_def.second));
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
        throw compilation_failed{fmt::format("glslangValidator command failed:\n{}\n\n{}\n\n{}",
                                             process.cout, process.cerr, fmt::join(command, " "))};
    }

    std::vector<uint32_t> spv = FileLoader::load_file<uint32_t>(output_file);

    std::filesystem::remove(output_file);

    return spv;
}

bool SystemGlslangValidatorCompiler::available() const {
    return !subprocess::find_program("glslangValidator").empty();
}

} // namespace merian
