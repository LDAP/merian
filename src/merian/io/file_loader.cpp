#include "merian/io/file_loader.hpp"
#include "merian/utils/string.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>

namespace merian {

bool FileLoader::exists(const std::filesystem::path& path,
                        std::filesystem::file_status file_status) {
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

// Searches the file in cwd and search paths and returns the full
// path to the file. If the filename is relative parents are searched if configured.
std::optional<std::filesystem::path>
FileLoader::find_file(const std::filesystem::path& filename) const {
    if (exists(filename)) {
        return std::filesystem::weakly_canonical(filename);
    }
    if (search_parents && filename.is_relative()) {
        std::filesystem::path current = std::filesystem::current_path();
        while(true) {
            const std::filesystem::path full_path = current / filename;
            if (exists(full_path)) {
                return std::filesystem::weakly_canonical(full_path);
            }

            if (current.parent_path() == current) {
                break;
            }
            current = current.parent_path();
        };

    }
    for (const auto& path : search_paths) {
        const std::filesystem::path full_path = path / filename;
        if (exists(full_path))
            return std::filesystem::weakly_canonical(full_path);
    }

    SPDLOG_WARN("file {} not found in search paths", filename.string());
    return std::nullopt;
}

std::optional<std::filesystem::path>
FileLoader::find_file(const std::filesystem::path& filename,
                      const std::filesystem::path& relative_to_file_or_directory) const {
    const std::filesystem::path relative_to = std::filesystem::is_directory(filename)
                                                  ? relative_to_file_or_directory
                                                  : relative_to_file_or_directory.parent_path();
    return find_file(relative_to / filename);
}

std::optional<std::string>
FileLoader::find_and_load_file(const std::filesystem::path& filename) const {
    std::optional<std::filesystem::path> full_path = find_file(filename);
    if (!full_path.has_value()) {
        return std::nullopt;
    }
    return load_file(full_path.value());
}

std::optional<std::string>
FileLoader::find_and_load_file(const std::filesystem::path& filename,
                               const std::filesystem::path& relative_to_file_or_directory) const {
    std::optional<std::filesystem::path> full_path =
        find_file(filename, relative_to_file_or_directory);
    if (!full_path.has_value()) {
        return std::nullopt;
    }
    return load_file(full_path.value());
}

void FileLoader::add_search_path(const std::filesystem::path path) {
    search_paths.insert(std::filesystem::weakly_canonical(path));
}

bool FileLoader::remove_search_path(const std::filesystem::path path) {
    return search_paths.erase(std::filesystem::weakly_canonical(path)) > 0;
}

void FileLoader::set_search_parents(const bool search_parents) {
    this->search_parents = search_parents;
}

} // namespace merian
