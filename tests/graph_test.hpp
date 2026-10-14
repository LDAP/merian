#pragma once

#include <gtest/gtest.h>

#include "ui_properties.hpp"

#include "merian-graph/graph/graph.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian/utils/properties_json_load.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"

#include <set>
#include <string>

class GraphTest : public ::testing::Test {
  protected:
    inline static merian::ContextHandle context;
    inline static merian::ResourceAllocatorHandle allocator;

    // Call after registering the test node types.
    static void create_context(const std::string& application_name) {
        setenv("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "true", 0);
        setenv("VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC", "true", 0);
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

    // Enough runs for every reconnect a run requests.
    static void run(const merian::GraphHandle& graph, const uint32_t runs = 3) {
        for (uint32_t i = 0; i < runs; i++) {
            graph->run();
        }
        graph->wait();
    }

    static UIProperties draw(const merian::GraphHandle& graph) {
        UIProperties props;
        graph->properties(props);
        return props;
    }

    static void set_node_enabled(const merian::GraphHandle& graph,
                                 const std::string& identifier,
                                 const bool enabled) {
        merian::JSONLoadProperties props(
            nlohmann::json{{"nodes", {{identifier, {{"enabled", enabled}}}}}});
        graph->properties(props);
    }

    static void set_node_properties(const merian::GraphHandle& graph,
                                    const std::string& identifier,
                                    const nlohmann::json& properties) {
        merian::JSONLoadProperties props(
            nlohmann::json{{"nodes", {{identifier, {{"properties", properties}}}}}});
        graph->properties(props);
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

    static bool has_connection(const merian::GraphHandle& graph,
                               const std::string& src,
                               const std::string& connection) {
        return stored_connections(graph->store_to_json())
            .contains(fmt::format("{}.{}", src, connection));
    }
};
