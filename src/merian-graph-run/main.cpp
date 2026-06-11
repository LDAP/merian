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

#include <atomic>
#include <csignal>
#include <fstream>
#include <optional>

namespace {

std::atomic_bool stop{false};

// Async-signal-safe: only touch the atomic, never lock or log.
void signal_handler(const int /*signal*/) {
    stop.store(true);
}

void print_usage() {
    fmt::print("usage: merian-graph-run [options] [graph.json]\n"
               "  -log-level=<trace|debug|info|warn|error>\n"
               "  -plugin-path=<dir>            (repeatable)\n"
               "  -validation=<on|off|ifdebug>  Vulkan validation layers (default: ifdebug)\n"
               "  --<alias> <value>            override a config value declared in the graph's\n"
               "                          \"short_config\" map (alias -> JSON pointer)\n"
               "  -help\n");
}

struct Options {
    std::optional<std::filesystem::path> config;
    bool validation = merian::Context::IS_DEBUG_BUILD;
    std::vector<std::pair<std::string, std::string>> overrides;
};

std::optional<Options> parse(const std::vector<std::string>& args) {
    Options options;
    for (size_t i = 1; i < args.size(); i++) {
        const std::string& arg = args[i];
        if (arg == "-help" || arg == "--help" || arg == "-h") {
            print_usage();
            return std::nullopt;
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
        }
    }
    return options;
}

// Resolves each --alias through the config's "short_config" map (alias -> JSON pointer) and
// replaces the pointed-to value.
void apply_overrides(nlohmann::json& config,
                     const std::vector<std::pair<std::string, std::string>>& overrides) {
    const nlohmann::json short_config = config.value("short_config", nlohmann::json::object());
    for (const auto& [alias, value] : overrides) {
        if (!short_config.contains(alias)) {
            SPDLOG_ERROR("override --{} is not declared in the config's short_config map", alias);
            continue;
        }
        const auto pointer =
            nlohmann::json::json_pointer(short_config.at(alias).get<std::string>());
        config[pointer] = value;
        SPDLOG_DEBUG("override --{} -> {} = {}", alias, pointer.to_string(), value);
    }
    config.erase("short_config");
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
        return 0;
    }

    std::vector<std::string> context_extensions = {merian::MerianGraphExtension::name};
    if (options->validation) {
        context_extensions.push_back(merian::ExtensionVkValidationLayers::name);
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

    const merian::ContextHandle context = merian::Context::create({
        .context_extensions = context_extensions,
        .additional_search_paths = search_paths,
        .application_name = "merian-graph-run",
    });

    const auto alloc =
        context->get_context_extension<merian::ExtensionResources>()->resource_allocator();
    const merian::GraphHandle graph =
        context->get_context_extension<merian::MerianGraphExtension>()->create({context, alloc});

    if (options->config) {
        // Resolve bare names (e.g. a shipped example) through the file loader search paths.
        std::filesystem::path config_path = *options->config;
        if (!std::filesystem::exists(config_path)) {
            if (const auto found = context->get_file_loader()->find_file(config_path)) {
                config_path = *found;
            }
        }
        std::ifstream stream(config_path);
        if (!stream) {
            SPDLOG_ERROR("could not open config '{}'", config_path.string());
            return 1;
        }
        nlohmann::json config = nlohmann::json::parse(stream);
        apply_overrides(config, options->overrides);
        graph->load_from_json(config);
        // load_from_json carries no path, so seed the load/store controls explicitly.
        graph->set_store_path(config_path);
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
