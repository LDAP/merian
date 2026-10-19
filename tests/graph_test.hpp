#pragma once

#include <gtest/gtest.h>

#include "ui_properties.hpp"

#include "merian-graph/graph/graph.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian-graph/nodes/host_sink/host_sink.hpp"
#include "merian/utils/properties_json_load.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"

#include <fmt/format.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

class GraphTest : public ::testing::Test {
  protected:
    inline static merian::ContextHandle context;
    inline static merian::ResourceAllocatorHandle allocator;

    static void set_env_default(const char* name, const char* value) {
#ifdef _WIN32
        if (std::getenv(name) == nullptr) {
            _putenv_s(name, value);
        }
#else
        setenv(name, value, 0);
#endif
    }

    // Call after registering the test node types.
    static void create_context(const std::string& application_name) {
        set_env_default("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "true");
        set_env_default("VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC", "true");
        context = merian::Context::create({
            .context_extensions = {merian::MerianGraphExtension::name,
                                   merian::ExtensionVkValidationLayers::name},
            .application_name = application_name,
        });
        allocator =
            context->get_context_extension<merian::ExtensionResources>()->resource_allocator();
    }

    static void TearDownTestSuite() {
        context->get_device()->get_device().waitIdle();
        allocator.reset();
        context.reset();
    }

    static merian::GraphHandle create_graph() {
        return context->get_context_extension<merian::MerianGraphExtension>()->create(
            {.context = context, .resource_allocator = allocator});
    }

    static merian::GraphHandle create_graph(const merian::GraphDescription& description) {
        const merian::GraphHandle graph = create_graph();
        graph->from_description(description);
        return graph;
    }

    static merian::GraphHandle create_graph(const nlohmann::json& description) {
        return create_graph(merian::GraphDescription::from_json(description));
    }

    static merian::GraphDescription color_to_sink() {
        merian::GraphDescription description;
        description.add_node("Color", "color",
                             {{"color", {0.25, 0.5, 0.75, 1}},
                              {"format", "R32G32B32A32Sfloat"},
                              {"extent", {2, 1, 1}}});
        description.add_node("Host Image Sink", "sink");
        description.add_connection("color", "sink", "out", "src");
        return description;
    }

    static std::array<float, 8> read_texels(const merian::HostSink::Capture& capture) {
        const merian::MemoryAllocationHandle& memory = capture.copy.memory;
        std::array<float, 8> texels;
        memory->invalidate();
        std::memcpy(texels.data(), memory->map(), sizeof(texels));
        memory->unmap();
        return texels;
    }

    // Enough runs for every reconnect a run requests.
    static void run(const merian::GraphHandle& graph, const uint32_t runs = 3) {
        for (uint32_t i = 0; i < runs; i++) {
            graph->run();
        }
        graph->wait();
    }

    static UIProperties draw(const merian::GraphHandle& graph) {
        UIProperties props;
        graph->draw_ui(props);
        return props;
    }

    static void set_node_enabled(const merian::GraphHandle& graph,
                                 const std::string& identifier,
                                 const bool enabled) {
        merian::JSONLoadProperties props(nlohmann::json{{identifier, {{"enabled", enabled}}}});
        graph->node_properties(identifier, props);
    }

    static std::set<std::string> stored_connections(const nlohmann::json& description) {
        std::set<std::string> connections;
        for (const auto& [identifier, node] : description.at("nodes").items()) {
            if (node.contains("outputs")) {
                for (const nlohmann::json& connection : node.at("outputs")) {
                    connections.insert(
                        fmt::format("{}.{}", identifier, connection.get<std::string>()));
                }
            }
        }
        return connections;
    }

    static std::set<std::string> stored_connections(const merian::GraphHandle& graph) {
        return stored_connections(graph->to_description().to_json());
    }

    static bool has_connection(const merian::GraphHandle& graph,
                               const std::string& src,
                               const std::string& connection) {
        return stored_connections(graph).contains(fmt::format("{}.{}", src, connection));
    }
};
