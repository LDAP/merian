#include "utils/string_utils.hpp"

#include <filesystem>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <fstream>

namespace merian {

class FileLoader {

    static bool exists(const std::filesystem::path& path,
                       std::filesystem::file_status file_status = std::filesystem::file_status{}) {
        if (std::filesystem::status_known(file_status) ? std::filesystem::exists(file_status)
                                                       : std::filesystem::exists(path))
            return true;
        else
            return false;
    }

    static std::string load_file(const std::filesystem::path& path) {
        if (!exists(path)) {
            throw std::runtime_error{fmt::format("failed to load {} (does not exist)", std::string(path))};
        }

        // Open the stream to 'lock' the file.
        std::ifstream f(path, std::ios::in | std::ios::binary);
        const auto size = std::filesystem::file_size(path);

        std::string result(size, '\0');
        f.read(result.data(), size);

        SPDLOG_DEBUG("load {} of data from {}", format_size(size), std::string(path));

        return result;
    }

  public:
    FileLoader(std::vector<std::filesystem::path> search_paths = {"./"}) : search_paths(search_paths) {}

    // Searches the file in cwd and search paths and returns the full path to the file
    std::optional<std::filesystem::path> find_file(const std::filesystem::path& filename) {
        if (exists(filename)) {
            return filename;
        }
        for (const auto& path : search_paths) {
            std::filesystem::path full_path = path / filename;
            if (exists(full_path))
                return full_path;
        }

        SPDLOG_WARN("file {} not found in search paths", std::string(filename));
        return std::nullopt;
    }

    std::optional<std::string> find_and_load_file(const std::filesystem::path& filename) {
        std::optional<std::filesystem::path> full_path = find_file(filename);
        if (!full_path.has_value()) {
            return std::nullopt;
        }
        return load_file(full_path.value());
    }

  private:
    std::vector<std::filesystem::path> search_paths;
};

} // namespace merian
