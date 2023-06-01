#pragma once

#include <filesystem>
#include <fstream>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace merian {

class FileLoader {

    static bool exists(const std::filesystem::path& path,
                       std::filesystem::file_status file_status = std::filesystem::file_status{});

    static std::string load_file(const std::filesystem::path& path);

  public:
    FileLoader(std::vector<std::filesystem::path> search_paths = {"./"}) : search_paths(search_paths) {}

    // Searches the file in cwd and search paths and returns the full path to the file
    std::optional<std::filesystem::path> find_file(const std::filesystem::path& filename);

    std::optional<std::string> find_and_load_file(const std::filesystem::path& filename);

  private:
    std::vector<std::filesystem::path> search_paths;
};

} // namespace merian
