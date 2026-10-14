#include "graph_test.hpp"

#include "merian-graph/nodes/svgf/svgf.hpp"

using namespace merian;
using json = nlohmann::json;

namespace {

class SVGFTest : public GraphTest {
  protected:
    static void SetUpTestSuite() {
        create_context("test-svgf");
    }

    static GraphHandle load(const json& properties) {
        const GraphHandle graph = create_graph();
        graph->load_from_json(
            json{{"version", 4},
                 {"nodes", {{"svgf", {{"type", "Denoiser (SVGF)"}, {"properties", properties}}}}}});
        run(graph);
        return graph;
    }

    static std::set<std::string> children(const GraphHandle& graph) {
        std::set<std::string> identifiers;
        for (const std::string& identifier : graph->identifiers()) {
            if (identifier.starts_with("svgf:")) {
                identifiers.insert(identifier);
            }
        }
        return identifiers;
    }
};

} // namespace

// The iterations shape the subgraph, the filter parameters reach every level and the settings of
// the variance estimate and the TAA persist through the parent.
TEST_F(SVGFTest, SettingsReachTheChildren) {
    const GraphHandle graph = load({{"iterations", 3},
                                    {"filter", {{"depth", 20.0}}},
                                    {"variance estimate", {{"spatial bias", 4.0}}},
                                    {"taa", {{"alpha", 0.5}}}});
    EXPECT_EQ(children(graph),
              (std::set<std::string>{"svgf:variance estimate", "svgf:filter 0", "svgf:filter 1",
                                     "svgf:filter 2", "svgf:taa"}));
    const auto filter = graph->find_node_for_identifier_and_type<SVGFFilter>("svgf:filter 2");
    ASSERT_TRUE(filter);
    EXPECT_EQ(filter->dump_config().at("depth"), 20.0);

    const json stored = graph->store_to_json();
    EXPECT_EQ(stored.at("nodes").size(), 1);
    const json& properties = stored.at("nodes").at("svgf").at("properties");
    EXPECT_EQ(properties.at("variance estimate").at("spatial bias"), 4.0);
    EXPECT_EQ(properties.at("taa").at("alpha"), 0.5);

    set_node_properties(graph, "svgf", {{"iterations", 1}, {"filter", {{"depth", 30.0}}}});
    run(graph);
    EXPECT_EQ(children(graph),
              (std::set<std::string>{"svgf:variance estimate", "svgf:filter 0", "svgf:taa"}));
    EXPECT_EQ(graph->find_node_for_identifier_and_type<SVGFFilter>("svgf:filter 0")
                  ->dump_config()
                  .at("depth"),
              30.0);
}
