#include <gtest/gtest.h>

#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/node_registry.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian/utils/properties_json_dump.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

using namespace merian;

namespace {

class RangeCheckProperties : public JSONDumpProperties {
  public:
    bool config_float(const std::string& id,
                      float* value,
                      const std::string& desc,
                      const int components,
                      const float sensitivity,
                      const std::optional<float>& min,
                      const std::optional<float>& max) override {
        if (!(sensitivity > 0.f)) {
            violations.push_back(fmt::format("'{}': sensitivity {}", id, sensitivity));
        }
        check(id, value, components, min, max);
        return JSONDumpProperties::config_float(id, value, desc, components, sensitivity, min, max);
    }

    bool config_int(const std::string& id,
                    int32_t* value,
                    const std::string& desc,
                    const int components,
                    const std::optional<int32_t>& min,
                    const std::optional<int32_t>& max) override {
        check(id, value, components, min, max);
        return JSONDumpProperties::config_int(id, value, desc, components, min, max);
    }

    bool config_uint(const std::string& id,
                     uint32_t* value,
                     const std::string& desc,
                     const int components,
                     const std::optional<uint32_t>& min,
                     const std::optional<uint32_t>& max) override {
        check(id, value, components, min, max);
        return JSONDumpProperties::config_uint(id, value, desc, components, min, max);
    }

    std::vector<std::string> violations;

  private:
    template <typename T>
    void check(const std::string& id,
               const T* value,
               const int components,
               const std::optional<T>& min,
               const std::optional<T>& max) {
        for (int i = 0; i < components; i++) {
            if ((min && value[i] < *min) || (max && value[i] > *max)) {
                violations.push_back(fmt::format("'{}': {} outside [{}, {}]", id, value[i],
                                                 min ? fmt::to_string(*min) : "-",
                                                 max ? fmt::to_string(*max) : "-"));
            }
        }
    }
};

} // namespace

TEST(NodeProperties, DefaultsWithinRanges) {
    const ContextHandle context = Context::create({
        .context_extensions = {MerianGraphExtension::name},
        .application_name = "test-node-properties",
    });
    const auto alloc = context->get_context_extension<ExtensionResources>()->resource_allocator();
    const GraphHandle graph = context->get_context_extension<MerianGraphExtension>()->create(
        {.context = context, .resource_allocator = alloc});

    for (const std::string& name : NodeRegistry::get_instance().node_names()) {
        const NodeHandle node = graph->find_node_for_identifier(graph->add_node(name));
        RangeCheckProperties props;
        std::ignore = node->properties(props);
        EXPECT_TRUE(props.violations.empty())
            << name << ": " << fmt::format("{}", fmt::join(props.violations, ", "));
    }
}
