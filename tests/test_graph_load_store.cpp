#include <gtest/gtest.h>

#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/graph_description.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"

#include <nlohmann/json.hpp>

using namespace merian;
using json = nlohmann::json;

TEST(GraphLoadStore, CliRoundTrip) {
    const ContextHandle context = Context::create({
        .context_extensions = {MerianGraphExtension::name},
        .application_name = "test-graph-load-store",
    });
    const auto alloc = context->get_context_extension<ExtensionResources>()->resource_allocator();
    const GraphHandle graph = context->get_context_extension<MerianGraphExtension>()->create(
        {.context = context, .resource_allocator = alloc});

    const json config = json::parse(R"({
        "version": 4,
        "nodes": {},
        "cli": {
            "--": {"pointer": "/nodes/shell/properties/startup commands"},
            "scene": {"pointer": "/nodes/scene/properties/file", "required": true}
        }
    })");

    graph->from_description(GraphDescription::from_json(config));
    EXPECT_EQ(graph->to_description().to_json().at("cli"), config.at("cli"));

    graph->from_description(
        GraphDescription::from_json(json::parse(R"({"version": 4, "nodes": {}})")));
    EXPECT_FALSE(graph->to_description().to_json().contains("cli"));

    // node metadata only lives in the loaded description and must be carried through
    const json with_metadata = json::parse(R"({
        "version": 4,
        "nodes": {"color": {"type": "Color", "metadata": {"pos": [10, 20]}}}
    })");
    graph->from_description(GraphDescription::from_json(with_metadata));
    EXPECT_EQ(graph->to_description().to_json().at("nodes").at("color").at("metadata"),
              with_metadata.at("nodes").at("color").at("metadata"));
}

TEST(GraphLoadStore, NoVersionIsTheCurrentOne) {
    const json config = json::parse(R"({
        "nodes": {
            "color": {"type": "Color", "enabled": false, "outputs": ["out->a.src", "out->b.src"]},
            "a": {"type": "Blit"},
            "b": {"type": "Blit", "outputs": ["out->a.prev"]}
        }
    })");
    const GraphDescription description = GraphDescription::from_json(config);
    EXPECT_FALSE(description.get_nodes().at("color").enabled);
    EXPECT_EQ(description.get_nodes().at("a").node_type, "Blit");

    const json stored = description.to_json();
    EXPECT_EQ(stored.at("version"), 4);
    for (const std::string identifier : {"color", "b"}) {
        EXPECT_EQ(stored.at("nodes").at(identifier).at("outputs"),
                  config.at("nodes").at(identifier).at("outputs"));
    }
    EXPECT_FALSE(stored.at("nodes").at("a").contains("outputs"));
}

TEST(GraphLoadStore, UnsupportedVersion) {
    EXPECT_THROW(GraphDescription::from_json(json::parse(R"({"version": 3, "nodes": {}})")),
                 std::runtime_error);
}

TEST(GraphMerge, ObjectRecurse) {
    json base = json::parse(R"({"a": {"x": 1, "y": 2}, "b": 1})");
    GraphDescription::merge_into(base, json::parse(R"({"a": {"y": 9, "z": 3}, "c": 4})"));
    EXPECT_EQ(base, json::parse(R"({"a": {"x": 1, "y": 9, "z": 3}, "b": 1, "c": 4})"));
}

TEST(GraphMerge, NodesById) {
    json base = json::parse(R"({"nodes": {
        "render": {"properties": {"spp": 2, "len": 3}},
        "scene":  {"properties": {"file": "a"}}
    }})");
    GraphDescription::merge_into(base, json::parse(R"({"nodes": {
        "render": {"properties": {"spp": 8}},
        "extra":  {"properties": {"k": 1}}
    }})"));

    EXPECT_EQ(base["nodes"].size(), 3);
    EXPECT_EQ(base["nodes"]["render"]["properties"]["spp"], 8);
    EXPECT_EQ(base["nodes"]["render"]["properties"]["len"], 3);
    EXPECT_TRUE(base["nodes"].contains("scene"));
    EXPECT_EQ(base["nodes"]["extra"]["properties"]["k"], 1);
}

TEST(GraphMerge, ScalarAndPlainArrayReplace) {
    json scalar = json::parse(R"({"x": 1})");
    GraphDescription::merge_into(scalar, json::parse(R"({"x": [1, 2]})"));
    EXPECT_EQ(scalar["x"], json::parse("[1, 2]"));

    json arr = json::parse(R"({"a": [1, 2, 3]})");
    GraphDescription::merge_into(arr, json::parse(R"({"a": [9]})"));
    EXPECT_EQ(arr["a"], json::parse("[9]"));
}

TEST(GraphMerge, AppendPrefix) {
    json base = json::parse(R"({"nodes": {"gbuffer": {"outputs": ["a", "b"]}}})");
    GraphDescription::merge_into(
        base, json::parse(R"({"nodes": {"gbuffer": {"$+$outputs": ["b", "c"]}}})"));
    EXPECT_EQ(base["nodes"]["gbuffer"]["outputs"], json::parse(R"(["a", "b", "c"])"));

    // the appended-to key, and every object on the way to it, may be missing
    json empty = json::parse("{}");
    GraphDescription::merge_into(empty,
                                 json::parse(R"({"nodes": {"gbuffer": {"$+$outputs": ["a"]}}})"));
    EXPECT_EQ(empty, json::parse(R"({"nodes": {"gbuffer": {"outputs": ["a"]}}})"));
}

TEST(GraphDescription, ConnectionReplacesTheInputsSource) {
    GraphDescription description;
    description.add_node("Color", "a");
    description.add_node("Color", "b");
    description.add_node("Blit", "blit");
    description.add_connection("a", "blit", "out", "src");
    description.add_connection("b", "blit", "out", "src");
    EXPECT_FALSE(description.remove_connection("a", "blit", "src"));
    EXPECT_FALSE(description.to_json().at("nodes").at("a").contains("outputs"));
    EXPECT_TRUE(description.remove_node("b"));
    EXPECT_TRUE(description.get_nodes().at("blit").incoming_connections.empty());
}

TEST(GraphDescription, NamesWithDashesAndDotsRoundTrip) {
    GraphDescription description;
    description.add_node("Color", "a-b");
    description.add_node("Blit", "c-d");
    description.add_connection("a-b", "c-d", "out-1.x", "src-2.y");
    const GraphDescription loaded = GraphDescription::from_json(description.to_json());
    EXPECT_EQ(loaded.get_nodes().at("c-d").incoming_connections.at("src-2.y"),
              (std::pair<std::string, std::string>{"a-b", "out-1.x"}));
    EXPECT_EQ(loaded.to_json(), description.to_json());
}

TEST(GraphDescription, GeneratedIdentifiersMatchTheGraphs) {
    GraphDescription description;
    EXPECT_EQ(description.add_node("Color"), "Color 0");
    EXPECT_EQ(description.add_node("Color"), "Color 1");
    EXPECT_EQ(description.add_node("Color", "Color 2"), "Color 2");
    EXPECT_EQ(description.add_node("Color"), "Color 3");

    const ContextHandle context = Context::create({
        .context_extensions = {MerianGraphExtension::name},
        .application_name = "test-graph-load-store",
    });
    const auto alloc = context->get_context_extension<ExtensionResources>()->resource_allocator();
    const GraphHandle graph = context->get_context_extension<MerianGraphExtension>()->create(
        {.context = context, .resource_allocator = alloc});
    EXPECT_EQ(graph->add_node("Color"), "Color 0");
    EXPECT_EQ(graph->add_node("Color"), "Color 1");
}

TEST(GraphDescription, ConnectionNeedsNodes) {
    GraphDescription description;
    description.add_node("Color", "color");
    EXPECT_THROW(description.add_connection("color", "missing", "out", "src"),
                 std::invalid_argument);
    EXPECT_THROW(description.add_connection("missing", "color", "out", "src"),
                 std::invalid_argument);
}
