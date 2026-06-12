#include <gtest/gtest.h>

#include "merian-graph/graph/graph.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"

#include <nlohmann/json.hpp>

using namespace merian;

TEST(GraphLoadStore, ShortConfigRoundTrip) {
    const ContextHandle context = Context::create({
        .context_extensions = {MerianGraphExtension::name},
        .application_name = "test-graph-load-store",
    });
    const auto alloc = context->get_context_extension<ExtensionResources>()->resource_allocator();
    const GraphHandle graph = context->get_context_extension<MerianGraphExtension>()->create(
        {.context = context, .resource_allocator = alloc});

    const nlohmann::json config = nlohmann::json::parse(R"({
        "version": 3,
        "nodes": [],
        "graph_properties": {
            "short_config": {
                "--": {"pointer": "/nodes/0/properties/startup commands", "required": false},
                "scene": {"pointer": "/nodes/1/properties/file", "required": true}
            }
        }
    })");

    graph->load_from_json(config);
    const nlohmann::json stored = graph->store_to_json();
    EXPECT_EQ(stored.at("graph_properties").at("short_config"),
              config.at("graph_properties").at("short_config"));

    // Loading a config without short configs must drop the previous ones.
    graph->load_from_json(nlohmann::json::parse(R"({"version": 3, "nodes": []})"));
    EXPECT_FALSE(graph->store_to_json()
                     .value("graph_properties", nlohmann::json::object())
                     .contains("short_config"));
}
