#include "merian/plugin/plugins.hpp"

#include "merian/io/file_loader.hpp"
#include "merian/plugin/plugin_export.hpp"

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string_view>
#include <unordered_set>

namespace merian {

#if defined(_WIN32)
static constexpr char ENV_PATH_SEPARATOR = ';';
static constexpr const char* PLUGIN_EXTENSION = ".dll";
#elif defined(__APPLE__)
static constexpr char ENV_PATH_SEPARATOR = ':';
static constexpr const char* PLUGIN_EXTENSION = ".dylib";
#else
static constexpr char ENV_PATH_SEPARATOR = ':';
static constexpr const char* PLUGIN_EXTENSION = ".so";
#endif

namespace {

std::vector<std::filesystem::path>& extra_search_paths() {
    static std::vector<std::filesystem::path> paths;
    return paths;
}

void append_env_paths(const char* env_var, std::vector<std::filesystem::path>& out) {
    const char* value = std::getenv(env_var);
    if (value == nullptr) {
        return;
    }
    std::string_view rest{value};
    while (!rest.empty()) {
        const auto sep = rest.find(ENV_PATH_SEPARATOR);
        const std::string_view part = rest.substr(0, sep);
        if (!part.empty()) {
            out.emplace_back(part);
        }
        if (sep == std::string_view::npos) {
            break;
        }
        rest.remove_prefix(sep + 1);
    }
}

std::optional<std::filesystem::path> user_plugin_dir() {
#if defined(_WIN32)
    if (const char* appdata = std::getenv("APPDATA")) {
        return std::filesystem::path(appdata) / "merian" / "plugins";
    }
#else
    if (const char* xdg = std::getenv("XDG_DATA_HOME")) {
        return std::filesystem::path(xdg) / "merian" / "plugins";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".local" / "share" / "merian" / "plugins";
    }
#endif
    return std::nullopt;
}

bool is_plugin_filename(const std::filesystem::path& path) {
    if (path.extension() != PLUGIN_EXTENSION) {
        return false;
    }
    std::string stem = path.stem().string();
    // Tolerate the platform "lib" prefix that some toolchains add.
    if (stem.starts_with("lib")) {
        stem.erase(0, 3);
    }
    return stem.starts_with("merian-plugin-");
}

} // namespace

void Plugins::add_search_path(const std::filesystem::path& path) {
    extra_search_paths().push_back(path);
}

std::vector<std::filesystem::path> Plugins::search_paths() {
    std::vector<std::filesystem::path> paths = extra_search_paths();

    append_env_paths("MERIAN_PLUGIN_PATH", paths);

    const std::filesystem::path exe_dir = FileLoader::binary_path().parent_path();
    paths.push_back(exe_dir);
    paths.push_back(exe_dir / "plugins");
    paths.push_back(exe_dir.parent_path() / "lib" / "merian" / "plugins");

    if (const auto user_dir = user_plugin_dir()) {
        paths.push_back(*user_dir);
    }

    return paths;
}

std::vector<std::filesystem::path> Plugins::discover() {
    std::vector<std::filesystem::path> result;
    std::unordered_set<std::string> seen;

    const auto scan = [&](const std::filesystem::path& dir) {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) {
            return;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (!entry.is_regular_file(ec) || !is_plugin_filename(entry.path())) {
                continue;
            }
            std::filesystem::path canonical = std::filesystem::weakly_canonical(entry.path());
            if (seen.insert(canonical.string()).second) {
                result.emplace_back(std::move(canonical));
            }
        }
    };

    for (const auto& dir : search_paths()) {
        scan(dir);
    }

    // A plugin cloned into <merian>/plugins/<name> is built as a merian subproject, landing in
    // <build>/subprojects/<name>/ next to the host binary.
    const std::filesystem::path exe_dir = FileLoader::binary_path().parent_path();
    const std::filesystem::path plugins_src = exe_dir.parent_path() / "plugins";
    std::error_code ec;
    if (std::filesystem::is_directory(plugins_src, ec)) {
        for (const auto& clone : std::filesystem::directory_iterator(plugins_src, ec)) {
            if (clone.is_directory(ec)) {
                scan(exe_dir / "subprojects" / clone.path().filename());
            }
        }
    }

    return result;
}

std::vector<std::filesystem::path> Plugins::resource_search_paths() {
    std::vector<std::filesystem::path> paths;
    for (const auto& plugin : discover()) {
        const std::filesystem::path dir = plugin.parent_path();
        std::string name = plugin.stem().string();
        if (name.starts_with("lib")) {
            name.erase(0, 3);
        }
        paths.push_back(dir / name);                // installed: resources next to the .so
        paths.push_back(dir.parent_path() / "res"); // standalone build: <build>/../res

        // Built as a merian subproject (<build>/subprojects/<clone>/...): resources stay in the
        // source tree at <merian>/plugins/<clone>/res.
        if (dir.parent_path().filename() == "subprojects") {
            paths.push_back(dir.parent_path().parent_path().parent_path() / "plugins" /
                            dir.filename() / "res");
        }
    }
    return paths;
}

void* validate_plugin_api(const DynamicLibrary& lib,
                          const char* register_symbol,
                          const char* abi_symbol,
                          std::uint32_t expected_abi,
                          std::string& out_name) {
    void* register_fn = lib.get_symbol(register_symbol);
    if (register_fn == nullptr) {
        return nullptr; // does not implement this API
    }

    const auto abi_version = lib.get_symbol<std::uint32_t (*)()>(abi_symbol);
    if (abi_version == nullptr) {
        SPDLOG_WARN("ignoring plugin '{}': exports {} but no {}", lib.get_path().string(),
                    register_symbol, abi_symbol);
        return nullptr;
    }
    if (abi_version() != expected_abi) {
        SPDLOG_WARN("ignoring plugin '{}': {} reports {} != host {}", lib.get_path().string(),
                    abi_symbol, abi_version(), expected_abi);
        return nullptr;
    }

    const auto name_fn = lib.get_symbol<merian_plugin_name_fn>("merian_plugin_name");
    out_name = name_fn != nullptr ? name_fn() : lib.get_path().stem().string();
    return register_fn;
}

void load_plugins(const char* register_symbol,
                  const char* abi_symbol,
                  const std::uint32_t expected_abi,
                  const std::string_view kind,
                  const std::function<void(void*)>& invoke) {
    static std::vector<DynamicLibrary> kept_alive;

    std::unordered_set<std::string> loaded_names;
    for (const auto& path : Plugins::discover()) {
        try {
            DynamicLibrary lib{path};
            std::string name;
            void* register_sym =
                validate_plugin_api(lib, register_symbol, abi_symbol, expected_abi, name);
            if (register_sym == nullptr) {
                continue;
            }
            // A plugin reachable through more than one path (e.g. an in-tree and a standalone build)
            // must only register once.
            if (!loaded_names.insert(name).second) {
                SPDLOG_WARN("skipping duplicate {} plugin '{}' ({})", kind, name, path.string());
                continue;
            }
            invoke(register_sym);
            SPDLOG_INFO("loaded {} plugin '{}' ({})", kind, name, path.string());
            kept_alive.emplace_back(std::move(lib));
        } catch (const DynamicLibrary::load_failed& e) {
            SPDLOG_WARN("could not load plugin '{}': {}", path.string(), e.what());
        }
    }
}

} // namespace merian
