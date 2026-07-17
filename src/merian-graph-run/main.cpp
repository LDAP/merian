#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/graph_description.hpp"
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
    fmt::print(
        "usage: merian-graph-run [options] [graph.json [args...]]\n"
        "  --loglevel=<trace|debug|info|warn|error>\n"
        "  --plugin-path=<dir>           (repeatable)\n"
        "  --validation=<on|off|ifdebug> Vulkan validation layers (default: ifdebug)\n"
        "  --merge <file.json>           deep-merge a JSON file into the config (repeatable,\n"
        "                                last wins)\n"
        "  --<name> <value>              set an override declared in the graph's \"cli\" block;\n"
        "                                may appear before or after graph.json, in any order\n"
        "                                variant selections persist when the graph is stored;\n"
        "                                re-selecting re-applies the variant's preset\n"
        "  args...                       positional args and any unrecognized --flag, joined\n"
        "                                with spaces in order, feed the cli \"--\" override\n"
        "  --help                        with a graph.json: also lists its cli overrides\n");
}

struct Options {
    std::optional<std::filesystem::path> config;
    bool validation = merian::Context::IS_DEBUG_BUILD;
    bool help = false;
    // Non-runner tokens in command-line order; classified against the graph's cli block once
    // the config is loaded. A pre-config override's value is kept adjacent to its --name.
    std::vector<std::string> graph_args;
};

bool is_help_flag(const std::string& arg) {
    return arg == "--help" || arg == "-h";
}

std::optional<Options> parse(const std::vector<std::string>& args) {
    Options options;
    // -help anywhere wins, so `merian-graph-run graph.json -help` lists the cli overrides
    // instead of forwarding "-help" to the graph.
    options.help = std::ranges::any_of(args, is_help_flag);
    bool config_found = false;
    for (size_t i = 1; i < args.size(); i++) {
        const std::string& arg = args[i];
        if (is_help_flag(arg)) {
            continue;
        }
        // Runner flags are consumed in any position and never reach the graph.
        if (arg == "--validation" || arg == "--validation=on") {
            options.validation = true;
        } else if (arg == "--validation=off") {
            options.validation = false;
        } else if (arg == "--validation=ifdebug") {
            options.validation = merian::Context::IS_DEBUG_BUILD;
        } else if (arg.starts_with("--validation=")) {
            SPDLOG_ERROR("invalid --validation value '{}' (expected on/off/ifdebug)", arg);
            return std::nullopt;
        } else if (arg.starts_with("--loglevel=")) {
            spdlog::set_level(spdlog::level::from_str(arg.substr(arg.find('=') + 1)));
        } else if (arg.starts_with("--plugin-path=")) {
            merian::Plugins::add_search_path(arg.substr(arg.find('=') + 1));
        } else if (config_found) {
            options.graph_args.push_back(arg);
        } else if (arg.starts_with("--")) {
            // A named override before the config: keep its value adjacent so the config
            // (the first bare token) is still located correctly.
            options.graph_args.push_back(arg);
            if (i + 1 < args.size()) {
                options.graph_args.push_back(args[++i]);
            }
        } else if (arg.starts_with("-")) {
            SPDLOG_ERROR("unknown option '{}'", arg);
            print_usage();
            return std::nullopt;
        } else {
            options.config = arg;
            config_found = true;
        }
    }
    return options;
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
        } catch (const nlohmann::json::exception& e) {
            SPDLOG_ERROR("could not parse config '{}': {}", config_path->string(), e.what());
            return 1;
        }
    } else if (!options->graph_args.empty()) {
        SPDLOG_ERROR("arguments given but no graph config");
        return 1;
    }

    std::vector<std::filesystem::path> cli_search_dirs;
    if (config_path) {
        cli_search_dirs.push_back(config_path->parent_path());
        cli_search_dirs.insert(cli_search_dirs.end(), search_paths.begin(), search_paths.end());
    }

    if (options->help) {
        print_usage();
        if (config_path) {
            fmt::print("{}", merian::GraphDescription::cli_help(config, cli_search_dirs));
        }
        return 0;
    }

    if (config_path) {
        if (!merian::GraphDescription::apply_cli(config, options->graph_args, cli_search_dirs)) {
            fmt::print("{}", merian::GraphDescription::cli_help(config, cli_search_dirs));
            return 1;
        }
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
