#include "merian/io/file_loader.hpp"
#include "merian/utils/string.hpp"

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
        throw std::runtime_error{fmt::format("failed to load {} (does not exist)", path.string())};
    }

    // Open the stream to 'lock' the file.
    std::ifstream f(path, std::ios::in | std::ios::binary);
    const auto size = std::filesystem::file_size(path);

    std::string result(size, '\0');
    f.read(result.data(), size);

    SPDLOG_DEBUG("load {} of data from {}", format_size(size), path.string());

    return result;
}

// Searches the file in cwd and search paths and returns the full path to the file
std::optional<std::filesystem::path> FileLoader::find_file(const std::filesystem::path& filename) const {
    if (exists(filename)) {
        return std::filesystem::weakly_canonical(filename);
    }
    for (const auto& path : search_paths) {
        std::filesystem::path full_path = path / filename;
        if (exists(full_path))
            return std::filesystem::weakly_canonical(full_path);
    }

    SPDLOG_WARN("file {} not found in search paths", filename.string());
    return std::nullopt;
}

std::optional<std::string> FileLoader::find_and_load_file(const std::filesystem::path& filename) const {
    std::optional<std::filesystem::path> full_path = find_file(filename);
    if (!full_path.has_value()) {
        return std::nullopt;
    }
    return load_file(full_path.value());
}

void FileLoader::add_search_path(const std::filesystem::path path) {
    search_paths.push_back(path);
}

bool FileLoader::remove_search_path(const std::filesystem::path path) {
    bool found = false;
    for (std::size_t i = 0; i < search_paths.size(); i++) {
        if (search_paths[i] == path) {
            found = true;
            std::swap(search_paths[i], search_paths[search_paths.size() - 1]);
            search_paths.pop_back();
        }
    }
    return found;
}

} // namespace merian
