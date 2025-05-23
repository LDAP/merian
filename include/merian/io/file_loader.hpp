#pragma once

#include "merian/utils/string.hpp"
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <spdlog/spdlog.h>
#include <string>

namespace merian {

class FileLoader {

  public:
    static bool exists(const std::filesystem::path& path,
                       std::filesystem::file_status file_status = std::filesystem::file_status{});

    template <typename T> static std::vector<T> load_file(const std::filesystem::path& path) {
        if (!exists(path)) {
            throw std::runtime_error{
                fmt::format("failed to load {} (does not exist)", path.string())};
        }

        // Open the stream to 'lock' the file.
        std::ifstream f(path, std::ios::in | std::ios::binary);
        const std::size_t size = std::filesystem::file_size(path);

        if (size % sizeof(T)) {
            SPDLOG_WARN("loading {} B of data into a vector quantized to {} B", size, sizeof(T));
        }

        std::vector<T> result((size + sizeof(T) - 1) / sizeof(T), {});
        f.read((char*)result.data(), (std::streamsize)size);

        SPDLOG_DEBUG("load {} of data from {}", format_size(size), path.string());

        return result;
    }

    static std::string load_file(const std::filesystem::path& path);

    static std::optional<std::filesystem::path>
    search_cwd_parents(const std::filesystem::path& path);

  public:
    FileLoader(const std::set<std::filesystem::path>& search_paths = {"./"})
        : search_paths(search_paths) {}

    // Searches the file in cwd and search paths and returns the full path to the file
    std::optional<std::filesystem::path> find_file(const std::filesystem::path& path) const;

    std::optional<std::filesystem::path>
    find_file(const std::filesystem::path& filename,
              const std::filesystem::path& relative_to_file_or_directory) const;

    std::optional<std::string> find_and_load_file(const std::filesystem::path& filename) const;

    std::optional<std::string>
    find_and_load_file(const std::filesystem::path& filename,
                       const std::filesystem::path& relative_to_file_or_directory) const;

    // adds the path to the file loader. The path is searched using the file loader, that means it
    // can be relative to any previously added search path.
    void add_search_path(const std::filesystem::path& path);

    void add_search_path(const std::vector<std::filesystem::path>& path);

    void add_search_path(const std::vector<std::string>& path);

    bool remove_search_path(const std::filesystem::path& path);

    // Search in parents of cwd
    void set_cwd_search_parents(const bool search_parents);

  private:
    std::set<std::filesystem::path> search_paths;
    bool enable_search_cwd_parents = true;
};
using FileLoaderHandle = std::shared_ptr<FileLoader>;

} // namespace merian
