#include "merian/io/file_loader.hpp"
#include "merian/utils/string.hpp"

#include <filesystem>
#include <fmt/ranges.h>
#include <fstream>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>

namespace merian {

bool FileLoader::exists(const std::filesystem::path& path,
                        std::filesystem::file_status file_status) {
    return (std::filesystem::status_known(file_status) ? std::filesystem::exists(file_status)
                                                       : std::filesystem::exists(path));
}

std::string FileLoader::load_file(const std::filesystem::path& path) {
    if (!exists(path)) {
        throw std::runtime_error{fmt::format("failed to load {} (does not exist)", path.string())};
    }

    // Open the stream to 'lock' the file.
    std::ifstream f(path, std::ios::in | std::ios::binary);
    const std::size_t size = std::filesystem::file_size(path);

    std::string result(size, '\0');
    f.read(result.data(), (std::streamsize)size);

    SPDLOG_DEBUG("load {} of data from {}", format_size(size), path.string());

    return result;
}

std::filesystem::path FileLoader::install_prefix() {
    const std::filesystem::path test_file =
        std::filesystem::path(MERIAN_INCLUDE_DIR_NAME) / std::filesystem::path(MERIAN_PROJECT_NAME);

    std::filesystem::path prefix = std::filesystem::path(MERIAN_INSTALL_PREFIX);
    if (FileLoader::exists(prefix / test_file)) {
        return prefix;
    }

    return std::filesystem::current_path();
}

std::filesystem::path FileLoader::install_includedir() {
    return install_prefix() / std::filesystem::path(MERIAN_INCLUDE_DIR_NAME);
}

std::filesystem::path FileLoader::install_datadir() {
    return install_prefix() / std::filesystem::path(MERIAN_DATA_DIR_NAME);
}

// returns empty path if not found.
std::optional<std::filesystem::path>
FileLoader::search_cwd_parents(const std::filesystem::path& path) {
    std::filesystem::path current = std::filesystem::current_path();
    while (true) {
        const std::filesystem::path full_path = current / path;
        if (exists(full_path)) {
            return std::filesystem::weakly_canonical(full_path);
        }
        if (current.parent_path() == current) {
            break;
        }
        current = current.parent_path();
    };

    return std::nullopt;
}

// Searches the file in cwd and search paths and returns the full
// path to the file. If the filename is relative parents are searched if configured.
std::optional<std::filesystem::path>
FileLoader::find_file(const std::filesystem::path& path) const {
    if (exists(path)) {
        return std::filesystem::weakly_canonical(path);
    }
    if (enable_search_cwd_parents && path.is_relative()) {
        const auto match_in_parents = search_cwd_parents(path);
        if (match_in_parents) {
            return *match_in_parents;
        }
    }
    for (const auto& search_path : search_paths) {
        const std::filesystem::path full_path = search_path / path;
        if (exists(full_path))
            return std::filesystem::weakly_canonical(full_path);
    }

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
    std::vector<std::string> str_search_paths;
    str_search_paths.reserve(search_paths.size());
    for (const auto& search_path : search_paths) {
        str_search_paths.emplace_back(search_path.string());
    }
    SPDLOG_DEBUG("file {} not found in search paths [{}]", path.string(),
                 fmt::join(str_search_paths, ", "));
#endif

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

void FileLoader::add_search_path(const std::filesystem::path& path) {
    auto resolved = find_file(path);
    if (!resolved) {
        resolved = std::filesystem::weakly_canonical(path);
    }
    search_paths.insert(*resolved);
    SPDLOG_DEBUG("added search path {}", resolved->string());
}

void FileLoader::add_search_path(const std::vector<std::filesystem::path>& paths) {
    for (const std::filesystem::path& path : paths) {
        add_search_path(path);
    }
}

void FileLoader::add_search_path(const std::vector<std::string>& paths) {
    for (const std::string& path : paths) {
        add_search_path(path);
    }
}

bool FileLoader::remove_search_path(const std::filesystem::path& path) {
    return search_paths.erase(std::filesystem::weakly_canonical(path)) > 0;
}

void FileLoader::set_cwd_search_parents(const bool search_parents) {
    this->enable_search_cwd_parents = search_parents;
}

} // namespace merian
