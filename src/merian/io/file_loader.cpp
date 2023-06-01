#include "merian/io/file_loader.hpp"
#include "merian/utils/string_utils.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace merian {

bool FileLoader::exists(const std::filesystem::path& path, std::filesystem::file_status file_status) {
    if (std::filesystem::status_known(file_status) ? std::filesystem::exists(file_status)
                                                   : std::filesystem::exists(path))
        return true;
    else
        return false;
}

std::string FileLoader::load_file(const std::filesystem::path& path) {
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

// Searches the file in cwd and search paths and returns the full path to the file
std::optional<std::filesystem::path> FileLoader::find_file(const std::filesystem::path& filename) {
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

std::optional<std::string> FileLoader::find_and_load_file(const std::filesystem::path& filename) {
    std::optional<std::filesystem::path> full_path = find_file(filename);
    if (!full_path.has_value()) {
        return std::nullopt;
    }
    return load_file(full_path.value());
}

} // namespace merian
