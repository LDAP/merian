#include "merian-graph/graph/graph.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian-graph/nodes/window/window_node.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/plugin/plugins.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <fstream>
#include <map>
#include <optional>
#include <utility>

namespace {

std::atomic_bool stop{false};

// Async-signal-safe: only touch the atomic, never lock or log.
void signal_handler(const int /*signal*/) {
    stop.store(true);
}

void print_usage() {
    fmt::print(
        "usage: merian-graph-run [options] [graph.json [args...]]\n"
        "  -log-level=<trace|debug|info|warn|error>\n"
        "  -plugin-path=<dir>            (repeatable)\n"
        "  -validation=<on|off|ifdebug>  Vulkan validation layers (default: ifdebug)\n"
        "  --<alias> <value>             set a config value declared in the graph's\n"
        "                                \"short_config\" map\n"
        "  args...                       everything after graph.json, joined with spaces,\n"
        "                                sets the short config \"--\" (none: value unchanged)\n"
        "  -help                         with a graph.json: also lists its short configs\n");
}

struct Options {
    std::optional<std::filesystem::path> config;
    bool validation = merian::Context::IS_DEBUG_BUILD;
    bool help = false;
    std::vector<std::pair<std::string, std::string>> overrides;
};

bool is_help_flag(const std::string& arg) {
    return arg == "-help" || arg == "--help" || arg == "-h";
}

std::optional<Options> parse(const std::vector<std::string>& args) {
    Options options;
    // -help anywhere wins, so `merian-graph-run graph.json -help` lists the short configs
    // instead of forwarding "-help" to the graph.
    options.help = std::ranges::any_of(args, is_help_flag);
    for (size_t i = 1; i < args.size(); i++) {
        const std::string& arg = args[i];
        if (is_help_flag(arg)) {
            continue;
        }
        if (arg == "-validation" || arg == "-validation=on") {
            options.validation = true;
        } else if (arg == "-validation=off") {
            options.validation = false;
        } else if (arg == "-validation=ifdebug") {
            options.validation = merian::Context::IS_DEBUG_BUILD;
        } else if (arg.starts_with("-validation=")) {
            SPDLOG_ERROR("invalid -validation value '{}' (expected on/off/ifdebug)", arg);
            return std::nullopt;
        } else if (arg.starts_with("-log-level=")) {
            spdlog::set_level(spdlog::level::from_str(arg.substr(arg.find('=') + 1)));
        } else if (arg.starts_with("-plugin-path=")) {
            merian::Plugins::add_search_path(arg.substr(arg.find('=') + 1));
        } else if (arg.starts_with("--")) {
            if (i + 1 >= args.size()) {
                SPDLOG_ERROR("missing value for override '{}'", arg);
                return std::nullopt;
            }
            options.overrides.emplace_back(arg.substr(2), args[++i]);
        } else if (arg.starts_with("-")) {
            SPDLOG_ERROR("unknown option '{}'", arg);
            print_usage();
            return std::nullopt;
        } else {
            options.config = arg;
            // Everything after the config goes verbatim into the "--" short config.
            if (i + 1 < args.size()) {
                std::string joined = args[i + 1];
                for (size_t j = i + 2; j < args.size(); j++) {
                    joined += ' ';
                    joined += args[j];
                }
                options.overrides.emplace_back("--", std::move(joined));
            }
            break;
        }
    }
    return options;
}

// A list of constant writes: JSON pointer -> literal value, applied verbatim.
using Patch = std::vector<std::pair<nlohmann::json::json_pointer, nlohmann::json>>;

struct ShortConfig {
    enum class Kind : uint8_t { VALUE, FLAG, VARIANT };

    std::string alias;
    Kind kind = Kind::VALUE;
    bool required = false;

    // Value: pointer receives the CLI value, set is applied alongside.
    nlohmann::json::json_pointer pointer;
    Patch set;

    // Flag: set on a truthy value, set_off on a falsy one.
    Patch set_off;

    // Variant: the CLI value selects a named patch; default is applied when the alias is absent.
    std::map<std::string, Patch> variants;
    std::optional<std::string> default_variant;
};

Patch parse_patch(const nlohmann::json& obj) {
    Patch patch;
    for (const auto& [pointer, value] : obj.items()) {
        patch.emplace_back(nlohmann::json::json_pointer(pointer), value);
    }
    return patch;
}

// Short configs are owned by the graph and live at graph_properties/short_config. An entry is one
// of (discriminated by "type", else inferred from its keys):
//   value:   {"pointer": "<ptr>", "set": {<ptr>: <const>, ...}}  - pointer takes the CLI value
//   flag:    {"set": {...}, "set_off": {...}}                    - --alias on|off picks a patch
//   variant: {"variants": {"name": {...}, ...}, "default": ...}  - --alias <name> picks a patch
std::vector<ShortConfig> parse_short_configs(const nlohmann::json& config) {
    std::vector<ShortConfig> entries;
    const nlohmann::json graph_properties =
        config.value("graph_properties", nlohmann::json::object());
    const nlohmann::json short_config =
        graph_properties.value("short_config", nlohmann::json::object());
    for (const auto& [alias, value] : short_config.items()) {
        ShortConfig entry;
        entry.alias = alias;
        entry.required = value.value("required", false);

        std::string type;
        if (value.contains("type")) {
            type = value.at("type").get<std::string>();
        } else if (value.contains("variants")) {
            type = "variant";
        } else if (value.contains("pointer")) {
            type = "value";
        } else {
            type = "flag";
        }

        if (type == "variant") {
            entry.kind = ShortConfig::Kind::VARIANT;
            const nlohmann::json variants = value.value("variants", nlohmann::json::object());
            for (const auto& [name, patch] : variants.items()) {
                entry.variants.emplace(name, parse_patch(patch));
            }
            if (value.contains("default")) {
                entry.default_variant = value.at("default").get<std::string>();
            }
        } else if (type == "flag") {
            entry.kind = ShortConfig::Kind::FLAG;
            entry.set = parse_patch(value.value("set", nlohmann::json::object()));
            entry.set_off = parse_patch(value.value("set_off", nlohmann::json::object()));
        } else {
            entry.kind = ShortConfig::Kind::VALUE;
            entry.pointer = nlohmann::json::json_pointer(value.at("pointer").get<std::string>());
            entry.set = parse_patch(value.value("set", nlohmann::json::object()));
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

void print_short_configs(const std::filesystem::path& config_path,
                         const std::vector<ShortConfig>& entries) {
    if (entries.empty()) {
        fmt::print("'{}' declares no short configs\n", config_path.string());
        return;
    }
    fmt::print("short configs of '{}':\n", config_path.string());
    for (const auto& entry : entries) {
        std::string usage;
        std::string target;
        switch (entry.kind) {
        case ShortConfig::Kind::VALUE:
            usage = entry.alias == "--" ? "args... (after graph.json)"
                                        : fmt::format("--{} <value>", entry.alias);
            target = entry.pointer.to_string();
            if (!entry.set.empty()) {
                target += fmt::format(" (+{} side effect(s))", entry.set.size());
            }
            break;
        case ShortConfig::Kind::FLAG:
            usage = fmt::format("--{} <on|off>", entry.alias);
            target = "flag";
            break;
        case ShortConfig::Kind::VARIANT: {
            std::string names;
            for (const auto& [name, _] : entry.variants) {
                names += names.empty() ? name : "|" + name;
            }
            usage = fmt::format("--{} <{}>", entry.alias, names);
            target =
                entry.default_variant ? fmt::format("default: {}", *entry.default_variant) : "";
            break;
        }
        }
        fmt::print("  {:<28} -> {}{}\n", usage, target, entry.required ? " (required)" : "");
    }
}

// Keeps the type of the value being replaced: non-string targets accept any JSON literal.
void set_value(nlohmann::json& config,
               const nlohmann::json::json_pointer& pointer,
               const std::string& value) {
    if (!config.contains(pointer) || !config.at(pointer).is_string()) {
        const nlohmann::json parsed = nlohmann::json::parse(value, nullptr, false);
        if (!parsed.is_discarded()) {
            config[pointer] = parsed;
            return;
        }
    }
    config[pointer] = value;
}

void apply_patch(nlohmann::json& config, const Patch& patch) {
    for (const auto& [pointer, value] : patch) {
        config[pointer] = value;
    }
}

// Returns false on an unrecognized variant/flag value (the override is otherwise applied).
bool apply_override(nlohmann::json& config, const ShortConfig& entry, const std::string& value) {
    switch (entry.kind) {
    case ShortConfig::Kind::VALUE:
        set_value(config, entry.pointer, value);
        apply_patch(config, entry.set);
        return true;
    case ShortConfig::Kind::FLAG: {
        std::string lower = value;
        std::ranges::transform(lower, lower.begin(),
                               [](unsigned char c) { return std::tolower(c); });
        if (lower.empty() || lower == "on" || lower == "true" || lower == "1" || lower == "yes") {
            apply_patch(config, entry.set);
        } else if (lower == "off" || lower == "false" || lower == "0" || lower == "no") {
            apply_patch(config, entry.set_off);
        } else {
            SPDLOG_ERROR("flag --{} expects on|off, got '{}'", entry.alias, value);
            return false;
        }
        return true;
    }
    case ShortConfig::Kind::VARIANT: {
        const auto it = entry.variants.find(value);
        if (it == entry.variants.end()) {
            std::string names;
            for (const auto& [name, _] : entry.variants) {
                names += names.empty() ? name : ", " + name;
            }
            SPDLOG_ERROR("unknown variant '{}' for --{} (expected: {})", value, entry.alias, names);
            return false;
        }
        apply_patch(config, it->second);
        return true;
    }
    }
    return true;
}

bool apply_overrides(nlohmann::json& config,
                     const std::vector<ShortConfig>& entries,
                     const std::vector<std::pair<std::string, std::string>>& overrides) {
    bool ok = true;
    std::vector<bool> provided(entries.size(), false);
    for (const auto& [alias, value] : overrides) {
        const auto it =
            std::ranges::find_if(entries, [&](const ShortConfig& e) { return e.alias == alias; });
        if (it == entries.end()) {
            SPDLOG_ERROR("override --{} is not declared in the config's short_config map", alias);
            ok = false;
            continue;
        }
        ok &= apply_override(config, *it, value);
        provided[it - entries.begin()] = true;
        SPDLOG_DEBUG("override {} = {}", alias == "--" ? alias : "--" + alias, value);
    }
    for (size_t i = 0; i < entries.size(); i++) {
        if (provided[i]) {
            continue;
        }
        if (entries[i].kind == ShortConfig::Kind::VARIANT && entries[i].default_variant) {
            const auto it = entries[i].variants.find(*entries[i].default_variant);
            if (it == entries[i].variants.end()) {
                SPDLOG_ERROR("default variant '{}' for --{} is not declared",
                             *entries[i].default_variant, entries[i].alias);
                ok = false;
            } else {
                apply_patch(config, it->second);
            }
        } else if (entries[i].required) {
            SPDLOG_ERROR("missing required override --{}", entries[i].alias);
            ok = false;
        }
    }
    return ok;
}

// Fallback when no config is given: a cleared window with the ImGui overlay.
void build_default_graph(const merian::GraphHandle& graph) {
    const std::string window = graph->add_node("Window", "window");
    const std::string imgui = graph->add_node("ImGui", "imgui");
    graph->add_connection(window, imgui, "acquire", "acquire");
    graph->add_connection(window, imgui, "window", "window");
    // Closing the window shuts the app down rather than removing the node.
    graph->find_node_for_identifier_and_type<merian::WindowNode>(window)->set_on_should_close(
        true, false, false);
    graph->set_store_path("merian-graph.json");
}

} // namespace

int main(const int argc, const char** argv) {
    const auto options = parse(std::vector<std::string>(argv, argv + argc));
    if (!options) {
        return 1;
    }

    std::vector<std::filesystem::path> search_paths;
    if (const auto dev_res = merian::FileLoader::search_cwd_parents("res")) {
        search_paths.push_back(*dev_res);
    }
    // So a bare config name (e.g. "imgui_window.json") resolves to a shipped example.
    if (const auto dev_examples = merian::FileLoader::search_cwd_parents("examples")) {
        search_paths.push_back(*dev_examples);
    }
    if (const auto prefix = merian::FileLoader::install_prefix()) {
        search_paths.push_back(*prefix / merian::FileLoader::install_datadir_name() / "merian" /
                               "examples");
    }

    std::optional<std::filesystem::path> config_path;
    nlohmann::json config;
    std::vector<ShortConfig> short_configs;
    if (options->config) {
        config_path = *options->config;
        if (!std::filesystem::exists(*config_path)) {
            if (const auto found = merian::FileLoader(search_paths).find_file(*config_path)) {
                config_path = *found;
            }
        }
        std::ifstream stream(*config_path);
        if (!stream) {
            SPDLOG_ERROR("could not open config '{}'", config_path->string());
            return 1;
        }
        try {
            config = nlohmann::json::parse(stream);
            short_configs = parse_short_configs(config);
        } catch (const nlohmann::json::exception& e) {
            SPDLOG_ERROR("could not parse config '{}': {}", config_path->string(), e.what());
            return 1;
        }
    } else if (!options->overrides.empty()) {
        SPDLOG_ERROR("overrides given but no graph config");
        return 1;
    }

    if (options->help) {
        print_usage();
        if (config_path) {
            print_short_configs(*config_path, short_configs);
        }
        return 0;
    }

    if (config_path && !apply_overrides(config, short_configs, options->overrides)) {
        print_short_configs(*config_path, short_configs);
        return 1;
    }

    std::vector<std::string> context_extensions = {merian::MerianGraphExtension::name};
    if (options->validation) {
        context_extensions.emplace_back(merian::ExtensionVkValidationLayers::name);
    }

    const merian::ContextHandle context = merian::Context::create({
        .context_extensions = context_extensions,
        .additional_search_paths = search_paths,
        .application_name = "merian-graph-run",
    });

    const auto alloc =
        context->get_context_extension<merian::ExtensionResources>()->resource_allocator();
    const merian::GraphHandle graph =
        context->get_context_extension<merian::MerianGraphExtension>()->create(
            {.context = context, .resource_allocator = alloc});

    if (config_path) {
        graph->load_from_json(config);
        // load_from_json carries no path, so seed the load/store controls explicitly.
        graph->set_store_path(*config_path);
    } else {
        SPDLOG_INFO("no graph config given; bootstrapping a default window graph");
        build_default_graph(graph);
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    while (!stop) {
        graph->run();
    }

    SPDLOG_INFO("shutting down");
    return 0;
}
