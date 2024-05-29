#include "shader_hotreloader.hpp"

namespace merian {

ShaderModuleHandle
HotReloader::get_shader(const std::filesystem::path& path,
                        const std::optional<vk::ShaderStageFlagBits> shader_kind) {
    std::optional<std::filesystem::path> canonical = std::filesystem::weakly_canonical(path);
    if (!canonical) {
        throw std::runtime_error{fmt::format("file not found {}", path.string())};
    }

    const std::filesystem::file_time_type last_write_time =
        std::filesystem::last_write_time(*canonical);
    if (!shaders.contains(*canonical)) {
        shaders[*canonical].shader = std::make_shared<ShaderModule>(
            context, compiler->compile_glsl(*canonical, shader_kind));
        shaders[*canonical].last_write_time = last_write_time;
    } else if (last_write_time > shaders[*canonical].last_write_time) {
        try {
            shaders[*canonical].shader = std::make_shared<ShaderModule>(
                context, compiler->compile_glsl(*canonical, shader_kind));
            shaders[*canonical].last_write_time = last_write_time;
        } catch (const ShaderCompiler::compilation_failed& e) {
            SPDLOG_WARN("compilation failed: {}", e.what());
        }
    } else {
        return shaders[*canonical].shader;
    }

    return shaders[*canonical].shader;
}

} // namespace merian
