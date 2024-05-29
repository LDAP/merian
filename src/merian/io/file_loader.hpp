#pragma once

#include <filesystem>
#include <optional>
#include <set>
#include <spdlog/spdlog.h>
#include <string>

namespace merian {

class FileLoader {

  public:
    static bool exists(const std::filesystem::path& path,
                       std::filesystem::file_status file_status = std::filesystem::file_status{});

    static std::string load_file(const std::filesystem::path& path);

  public:
    FileLoader(std::set<std::filesystem::path> search_paths = {"./"})
        : search_paths(search_paths) {}

    // Searches the file in cwd and search paths and returns the full path to the file
    std::optional<std::filesystem::path> find_file(const std::filesystem::path& filename) const;

    std::optional<std::filesystem::path>
    find_file(const std::filesystem::path& filename,
              const std::filesystem::path& relative_to_file_or_directory) const;

    std::optional<std::string> find_and_load_file(const std::filesystem::path& filename) const;

    std::optional<std::string>
    find_and_load_file(const std::filesystem::path& filename,
                       const std::filesystem::path& relative_to_file_or_directory) const;

    void add_search_path(const std::filesystem::path path);

    bool remove_search_path(const std::filesystem::path path);

  private:
    std::set<std::filesystem::path> search_paths;
};
using FileLoaderHandle = std::shared_ptr<FileLoader>;


} // namespace merian
