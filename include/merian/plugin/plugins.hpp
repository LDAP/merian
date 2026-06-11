#pragma once

#include "merian/utils/dynamic_library.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace merian {

// Discovers merian-plugin-* shared libraries across the search paths.
class Plugins {
  public:
    // Adds a directory to search ahead of the environment-derived ones.
    static void add_search_path(const std::filesystem::path& path);

    // Directories searched, in order: add_search_path(), $MERIAN_PLUGIN_PATH, the executable
    // directory and its plugin subdirs, the user data dir.
    static std::vector<std::filesystem::path> search_paths();

    // Canonical, deduplicated paths of all discovered plugin libraries, in search order.
    static std::vector<std::filesystem::path> discover();

    // Resource search paths contributed by the discovered plugins; nonexistent ones are harmless.
    static std::vector<std::filesystem::path> resource_search_paths();
};

// Validates one plugin API on lib (an exported register_symbol and an abi_symbol matching
// expected_abi), returning the register symbol and writing the plugin's name to out_name. Returns
// nullptr if the API is absent, or warns and returns nullptr on an ABI mismatch.
void* validate_plugin_api(const DynamicLibrary& lib,
                          const char* register_symbol,
                          const char* abi_symbol,
                          std::uint32_t expected_abi,
                          std::string& out_name);

// Loads every discovered plugin that exports register_symbol with a matching abi_symbol and passes
// its resolved register symbol to invoke. Kind labels the plugin API in log output. Loaded
// libraries are kept alive for the process lifetime.
void load_plugins(const char* register_symbol,
                  const char* abi_symbol,
                  std::uint32_t expected_abi,
                  std::string_view kind,
                  const std::function<void(void* register_symbol)>& invoke);

} // namespace merian
