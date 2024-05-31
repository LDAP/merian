#include "shader_hotreloader.hpp"

namespace merian {

ShaderModuleHandle
HotReloader::get_shader(const std::filesystem::path& path,
                        const std::optional<vk::ShaderStageFlagBits> shader_kind) {
    std::optional<std::filesystem::path> canonical = std::filesystem::weakly_canonical(path);
    if (!canonical) {
        throw std::runtime_error{fmt::format("file not found {}", path.string())};
    }

    using namespace std::chrono_literals;

    const std::filesystem::file_time_type last_write_time =
        std::filesystem::last_write_time(*canonical);

    if (!shaders.contains(*canonical) ||
        (clock_cast<std::chrono::file_clock>(std::chrono::system_clock::now() - 200ms) >
             last_write_time &&
         last_write_time > shaders[*canonical].last_write_time)) {
        // wait additional 200ms, else the write to the file might still be in process.

        per_path& path_info = shaders[*canonical];

        // still remember time, so that we do not attempt to recompile the same broken file over
        // and over again.
        path_info.last_write_time = last_write_time;
        try {
            path_info.shader = std::make_shared<ShaderModule>(
                context, compiler->compile_glsl(*canonical, shader_kind));
            path_info.error.reset();
        } catch (const ShaderCompiler::compilation_failed& e) {
            path_info.shader = nullptr;
            path_info.error = e;
            throw e;
        }
    }

    per_path& path_info = shaders[*canonical];
    if (path_info.error) {
        throw *path_info.error;
    }
    return path_info.shader;
}

} // namespace merian
