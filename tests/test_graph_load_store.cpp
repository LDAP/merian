#include <gtest/gtest.h>

#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/graph_description.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

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
        "version": 3,
        "nodes": [],
        "cli": {
            "--": {"pointer": "/nodes/0/properties/startup commands"},
            "scene": {"pointer": "/nodes/1/properties/file", "required": true}
        }
    })");

    graph->load_from_json(config);
    EXPECT_EQ(graph->store_to_json().at("cli"), config.at("cli"));

    graph->load_from_json(json::parse(R"({"version": 3, "nodes": []})"));
    EXPECT_FALSE(graph->store_to_json().contains("cli"));

    // node metadata only lives in the loaded description and must be carried through
    const json with_metadata = json::parse(R"({
        "version": 3,
        "nodes": [{"id": "color", "type": "Color", "metadata": {"pos": [10, 20]}}]
    })");
    graph->load_from_json(with_metadata);
    EXPECT_EQ(graph->store_to_json().at("nodes").at(0).at("metadata"),
              with_metadata.at("nodes").at(0).at("metadata"));
}

TEST(GraphCli, ValueWithSet) {
    json config = json::parse(R"({
        "nodes": [
            {"id": "render", "properties": {"samples per pixel": 2}},
            {"id": "scene",  "properties": {"file": "", "env": {"Type": "Empty", "Path": ""}}}
        ],
        "cli": {
            "--":      {"pointer": "/nodes/1/properties/file"},
            "env-map": {"pointer": "/nodes/1/properties/env/Path",
                        "set": {"/nodes/1/properties/env/Type": "LatLong"}},
            "spp":     {"pointer": "/nodes/0/properties/samples per pixel"}
        }
    })");

    EXPECT_TRUE(GraphDescription::apply_cli(
        config, {"--env-map", "/tmp/x.hdr", "--spp", "8", "/scene.gltf"}));

    EXPECT_EQ(config["nodes"][1]["properties"]["env"]["Path"], "/tmp/x.hdr");
    EXPECT_EQ(config["nodes"][1]["properties"]["env"]["Type"], "LatLong");
    EXPECT_EQ(config["nodes"][0]["properties"]["samples per pixel"], 8);
    EXPECT_EQ(config["nodes"][1]["properties"]["file"], "/scene.gltf");
}

TEST(GraphCli, Variant) {
    const json base = json::parse(R"({
        "v": 0,
        "cli": {"quality": {"type": "variant", "default": "high",
                            "variants": {"low": {"set": {"/v": 1}}, "high": {"set": {"/v": 8}}}}}
    })");

    json with_default = base;
    EXPECT_TRUE(GraphDescription::apply_cli(with_default, {}));
    EXPECT_EQ(with_default["v"], 8);

    json low = base;
    EXPECT_TRUE(GraphDescription::apply_cli(low, {"--quality", "low"}));
    EXPECT_EQ(low["v"], 1);

    json bad = base;
    EXPECT_FALSE(GraphDescription::apply_cli(bad, {"--quality", "ultra"}));
    EXPECT_EQ(bad["v"], 0);
}

TEST(GraphCli, Flag) {
    const json base = json::parse(R"({
        "d": false,
        "cli": {"denoise": {"type": "flag", "set": {"/d": true}, "off": {"set": {"/d": false}}}}
    })");

    json on = base;
    EXPECT_TRUE(GraphDescription::apply_cli(on, {"--denoise", "on"}));
    EXPECT_EQ(on["d"], true);

    json off = base;
    off["d"] = true;
    EXPECT_TRUE(GraphDescription::apply_cli(off, {"--denoise", "off"}));
    EXPECT_EQ(off["d"], false);
}

TEST(GraphCli, RequiredAndUnknown) {
    const json config = json::parse(R"({
        "nodes": [{"id": "scene", "properties": {"file": ""}}],
        "cli": {"scene": {"pointer": "/nodes/0/properties/file", "required": true}}
    })");

    json missing = config;
    EXPECT_FALSE(GraphDescription::apply_cli(missing, {}));

    json given = config;
    EXPECT_TRUE(GraphDescription::apply_cli(given, {"--scene", "/s.gltf"}));
    EXPECT_EQ(given["nodes"][0]["properties"]["file"], "/s.gltf");

    // a value a stored session already materialized satisfies the requirement
    EXPECT_TRUE(GraphDescription::apply_cli(given, {}));

    // An undeclared --flag is positional; with no "--" override declared, that's an error.
    json unknown = config;
    EXPECT_FALSE(GraphDescription::apply_cli(unknown, {"--nope", "x"}));
}

TEST(GraphCli, OrderLastWins) {
    json config = json::parse(R"({"v": 0, "cli": {"spp": {"pointer": "/v"}}})");
    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--spp", "1", "--spp", "2"}));
    EXPECT_EQ(config["v"], 2);
}

TEST(GraphCli, IdBasedPointer) {
    json config = json::parse(R"({
        "nodes": [
            {"id": "render", "properties": {"samples per pixel": 2}},
            {"id": "scene",  "properties": {"file": ""}}
        ],
        "cli": {
            "--":  {"pointer": "/nodes/scene/properties/file"},
            "spp": {"pointer": "/nodes/render/properties/samples per pixel"},
            "old": {"pointer": "/nodes/0/properties/samples per pixel"}
        }
    })");

    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--spp", "8", "/scene.gltf"}));
    EXPECT_EQ(config["nodes"][0]["properties"]["samples per pixel"], 8);
    EXPECT_EQ(config["nodes"][1]["properties"]["file"], "/scene.gltf");

    // numeric pointers keep working
    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--old", "4"}));
    EXPECT_EQ(config["nodes"][0]["properties"]["samples per pixel"], 4);

    json unknown = json::parse(R"({
        "nodes": [{"id": "render", "properties": {}}],
        "cli": {"spp": {"pointer": "/nodes/nope/properties/spp"}}
    })");
    EXPECT_FALSE(GraphDescription::apply_cli(unknown, {"--spp", "8"}));
}

TEST(GraphCli, MergeDeclaresOverride) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "merian-test-fragment.json";
    {
        std::ofstream out(file);
        out << R"({
            "nodes": [{"id": "render", "properties": {"samples per pixel": 2}}],
            "cli": {"spp": {"pointer": "/nodes/render/properties/samples per pixel"}}
        })";
    }

    // overrides declared by a merged fragment are recognized in the same invocation, in any
    // argument order, and the explicit value wins over the fragment's
    for (const auto& args : {std::vector<std::string>{"--merge", file.string(), "--spp", "8"},
                             std::vector<std::string>{"--spp", "8", "--merge", file.string()}}) {
        json config = json::parse(R"({"nodes": [{"id": "scene"}]})");
        EXPECT_TRUE(GraphDescription::apply_cli(config, args));
        EXPECT_EQ(config["nodes"][1]["properties"]["samples per pixel"], 8);
    }

    std::filesystem::remove(file);
}

TEST(GraphCli, DefaultVariantDeclaresOverride) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "merian-test-renderer.json";
    {
        std::ofstream out(file);
        out << R"({
            "nodes": [{"id": "render", "properties": {"samples per pixel": 2}}],
            "cli": {"spp": {"pointer": "/nodes/render/properties/samples per pixel"}}
        })";
    }

    json config = json::parse(R"({
        "nodes": [{"id": "scene"}],
        "cli": {"renderer": {"type": "variant", "default": "pt", "variants": {"pt": {}}}}
    })");
    config["cli"]["renderer"]["variants"]["pt"]["merge"] = file.string();

    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--spp", "8"}));
    EXPECT_EQ(config["nodes"][1]["properties"]["samples per pixel"], 8);
    EXPECT_EQ(config["cli"]["renderer"]["selected"], "pt");

    std::filesystem::remove(file);
}

TEST(GraphCli, VariantSticky) {
    const json base = json::parse(R"({
        "v": 0,
        "cli": {"quality": {"type": "variant", "default": "high",
                            "variants": {"low": {"set": {"/v": 1}}, "high": {"set": {"/v": 8}}}}}
    })");

    json config = base;
    EXPECT_TRUE(GraphDescription::apply_cli(config, {}));
    EXPECT_EQ(config["v"], 8);
    EXPECT_EQ(config["cli"]["quality"]["selected"], "high");

    // a persisted selection is not re-applied
    config["v"] = 42;
    EXPECT_TRUE(GraphDescription::apply_cli(config, {}));
    EXPECT_EQ(config["v"], 42);

    // an explicit selection always re-applies its patch
    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--quality", "high"}));
    EXPECT_EQ(config["v"], 8);

    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--quality", "low"}));
    EXPECT_EQ(config["v"], 1);
    EXPECT_EQ(config["cli"]["quality"]["selected"], "low");
}

TEST(GraphCli, PatchMergeThenSet) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "merian-test-merge-set.json";
    {
        std::ofstream out(file);
        out << R"({"nodes": [{"id": "render", "properties": {"max path length": 12}}]})";
    }

    // the set targets a node that only exists after the patch's own merge
    json config = json::parse(R"({
        "nodes": [{"id": "scene"}],
        "cli": {"renderer": {"type": "variant", "variants": {"pt": {
            "set": {"/nodes/render/properties/max path length": 3}}}}}
    })");
    config["cli"]["renderer"]["variants"]["pt"]["merge"] = file.string();

    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--renderer", "pt"}));
    EXPECT_EQ(config["nodes"][1]["properties"]["max path length"], 3);

    std::filesystem::remove(file);
}

TEST(GraphCli, NullRemovesOverride) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "merian-test-null.json";
    {
        std::ofstream out(file);
        out << R"({"cli": {"max-path-length": null}})";
    }

    json config = json::parse(R"({"v": 5, "cli": {"max-path-length": {"pointer": "/v"}}})");
    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--merge", file.string()}));
    EXPECT_FALSE(config["cli"].contains("max-path-length"));

    // the removed override is no longer recognized
    json again = config;
    EXPECT_FALSE(GraphDescription::apply_cli(again, {"--max-path-length", "3"}));

    std::filesystem::remove(file);
}

TEST(GraphCli, MergeFile) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "merian-test-merge.json";
    {
        std::ofstream out(file);
        out << R"({"nodes": [{"id": "render", "properties": {"samples per pixel": 16}}]})";
    }

    json config = json::parse(R"({
        "nodes": [{"id": "render", "properties": {"samples per pixel": 2, "max path length": 5}}]
    })");
    EXPECT_TRUE(GraphDescription::apply_cli(config, {"--merge", file.string()}));
    EXPECT_EQ(config["nodes"][0]["properties"]["samples per pixel"], 16);
    EXPECT_EQ(config["nodes"][0]["properties"]["max path length"], 5);

    std::filesystem::remove(file);
}

TEST(GraphMerge, ObjectRecurse) {
    json base = json::parse(R"({"a": {"x": 1, "y": 2}, "b": 1})");
    GraphDescription::merge_into(base, json::parse(R"({"a": {"y": 9, "z": 3}, "c": 4})"));
    EXPECT_EQ(base, json::parse(R"({"a": {"x": 1, "y": 9, "z": 3}, "b": 1, "c": 4})"));
}

TEST(GraphMerge, NodeArrayById) {
    json base = json::parse(R"({"nodes": [
        {"id": "render", "properties": {"spp": 2, "len": 3}},
        {"id": "scene",  "properties": {"file": "a"}}
    ]})");
    GraphDescription::merge_into(base, json::parse(R"({"nodes": [
        {"id": "render", "properties": {"spp": 8}},
        {"id": "extra",  "properties": {"k": 1}}
    ]})"));

    EXPECT_EQ(base["nodes"].size(), 3);
    EXPECT_EQ(base["nodes"][0]["properties"]["spp"], 8);
    EXPECT_EQ(base["nodes"][0]["properties"]["len"], 3);
    EXPECT_EQ(base["nodes"][1]["id"], "scene");
    EXPECT_EQ(base["nodes"][2]["id"], "extra");
}

TEST(GraphMerge, ScalarAndPlainArrayReplace) {
    json scalar = json::parse(R"({"x": 1})");
    GraphDescription::merge_into(scalar, json::parse(R"({"x": [1, 2]})"));
    EXPECT_EQ(scalar["x"], json::parse("[1, 2]"));

    json arr = json::parse(R"({"a": [1, 2, 3]})");
    GraphDescription::merge_into(arr, json::parse(R"({"a": [9]})"));
    EXPECT_EQ(arr["a"], json::parse("[9]"));
}
