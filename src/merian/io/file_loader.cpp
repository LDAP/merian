#include "merian/io/file_loader.hpp"
#include "merian/utils/string.hpp"

#include <filesystem>
#include <fmt/ranges.h>
#include <fstream>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>

#if defined(__APPLE__)
#include <libproc.h>
#endif

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#include <shlobj.h>
#endif

#if defined(__FreeBSD__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

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

std::filesystem::path FileLoader::binary_path() {
#ifdef __linux__
    return std::filesystem::canonical("/proc/self/exe");
#elif defined(__FreeBSD__)
    std::array<char, 1024> path;
    size_t len_procpath = path.size();
    int mib_procpath[] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
    sysctl(mib_procpath, 4, path.data(), &len_procpath, NULL, 0);
    return std::filesystem::canonical(path.data());
#elif defined(_WIN32)
    char* path;
    _get_pgmptr(&path);
    return std::filesystem::canonical(path);
#elif defined(__APPLE__)
    pid_t pid = getpid();
    std::array<char, 1024> path;
    proc_pidpath(pid, path.data(), path.size());
    return std::filesystem::canonical(path.data());
#else
#error "binary_path not implemented for this target system"
#endif
}

std::optional<std::filesystem::path> FileLoader::portable_prefix() {
    const std::filesystem::path test_file =
        std::filesystem::path(MERIAN_INCLUDE_DIR_NAME) / std::filesystem::path(MERIAN_PROJECT_NAME);

    std::optional<std::filesystem::path> portable_prefix =
        search_parents(binary_path().parent_path(), test_file);
    if (portable_prefix) {
        return portable_prefix;
    }

    portable_prefix = search_parents(std::filesystem::current_path(), test_file);
    return portable_prefix;
}

std::optional<std::filesystem::path> FileLoader::install_prefix() {
    const std::filesystem::path test_file =
        install_includedir_name() / std::filesystem::path(MERIAN_PROJECT_NAME);

    std::filesystem::path prefix(MERIAN_INSTALL_PREFIX);
    if (exists(prefix / test_file)) {
        return prefix;
    }
    return std::nullopt;
}

std::filesystem::path FileLoader::install_includedir_name() {
    return std::filesystem::path(MERIAN_INCLUDE_DIR_NAME);
}

std::filesystem::path FileLoader::install_datadir_name() {
    return std::filesystem::path(MERIAN_DATA_DIR_NAME);
}

std::optional<std::filesystem::path> FileLoader::search_parents(const std::filesystem::path& start,
                                                                const std::filesystem::path& test) {
    std::filesystem::path current = start;
    while (true) {
        const std::filesystem::path full_test_path = current / test;
        if (exists(full_test_path)) {
            return std::filesystem::weakly_canonical(current);
        }
        if (current.parent_path() == current) {
            break;
        }
        current = current.parent_path();
    };

    return std::nullopt;
}

// returns empty path if not found.
std::optional<std::filesystem::path>
FileLoader::search_cwd_parents(const std::filesystem::path& path) {
    const std::optional<std::filesystem::path> base =
        search_parents(std::filesystem::current_path(), path);
    if (base) {
        return std::filesystem::weakly_canonical(*base / path);
    }

    return std::nullopt;
}

// Searches the file in cwd and search paths and returns the full
// path to the file. If the filename is relative parents are searched if configured.
//
// Additional search paths are preferred.
std::optional<std::filesystem::path>
FileLoader::find_file(const std::filesystem::path& path,
                      const vk::ArrayProxy<std::filesystem::path>& additional_search_paths) const {
    if (exists(path)) {
        return std::filesystem::weakly_canonical(path);
    }
    if (enable_search_cwd_parents && path.is_relative()) {
        const auto match_in_parents = search_cwd_parents(path);
        if (match_in_parents) {
            return match_in_parents;
        }
    }

    for (const auto& search_path : additional_search_paths) {
        std::filesystem::path full_path = std::filesystem::weakly_canonical(
            (std::filesystem::is_directory(search_path) ? search_path : search_path.parent_path()) /
            path);
        if (exists(full_path))
            return full_path;
    }

    for (const auto& search_path : search_paths) {
        std::filesystem::path full_path = std::filesystem::weakly_canonical(search_path / path);
        if (exists(full_path))
            return full_path;
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
    if (resolved) {
        search_paths.insert(*resolved);
        SPDLOG_DEBUG("added search path {}", resolved->string());
        return;
    }

    SPDLOG_DEBUG("path {} could not be found in search path and was not added as new search path.",
                 path.string());
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
