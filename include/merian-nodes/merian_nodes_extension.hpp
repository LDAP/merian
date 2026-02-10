#pragma once

#include "merian-nodes/graph/graph.hpp"
#include "merian-nodes/graph/node_registry.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/extension_glsl_compiler.hpp"
#include "merian/vk/extension/extension_slang_compiler.hpp"

namespace merian {

class MerianNodesExtension : public ContextExtension {
  public:
    MerianNodesExtension() : ContextExtension() {}

    std::vector<std::string> request_extensions() override {
        std::vector<std::string> aggregated;

        // Request compiler extensions for nodes that need shader compilation
        aggregated.push_back("merian-glsl-compiler");
        aggregated.push_back("merian-slang-session");

        auto& registry = NodeRegistry::get_instance();
        for (const auto& type_name : registry.node_type_names()) {
            auto node = registry.create_node_from_type(type_name);
            insert_all(aggregated, node->request_context_extensions());
        }

        return aggregated;
    }

    InstanceSupportInfo
    query_instance_support(const InstanceSupportQueryInfo& query_info) override {
        InstanceSupportInfo aggregated;
        aggregated.supported = true;

        auto& registry = NodeRegistry::get_instance();
        for (const auto& type_name : registry.node_type_names()) {
            auto node = registry.create_node_from_type(type_name);
            auto support_info = node->query_instance_support(query_info);
            SPDLOG_DEBUG("node {} instance support: {}", type_name, support_info);

            insert_all(aggregated.required_extensions, support_info.required_extensions);
            insert_all(aggregated.required_layers, support_info.required_layers);
        }

        return aggregated;
    }

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override {
        DeviceSupportInfo aggregated = DeviceSupportInfo::check(query_info, {
                                                                                "timelineSemaphore",
                                                                                "hostQueryReset",
                                                                            });

        aggregated.supported &=
            query_info.extension_container.get_context_extension<ExtensionSlangCompiler>(true) !=
            nullptr;
        aggregated.supported &=
            query_info.extension_container.get_context_extension<ExtensionGLSLCompiler>(true) !=
            nullptr;

        auto& registry = NodeRegistry::get_instance();
        for (const auto& type_name : registry.node_type_names()) {
            auto node = registry.create_node_from_type(type_name);
            auto support_info = node->query_device_support(query_info);
            SPDLOG_DEBUG("node {} device support: {}", type_name, support_info);

            insert_all(aggregated.required_extensions, support_info.required_extensions);
            insert_all(aggregated.required_features, support_info.required_features);
            insert_all(aggregated.required_spirv_extensions,
                       support_info.required_spirv_extensions);
            insert_all(aggregated.required_spirv_capabilities,
                       support_info.required_spirv_capabilities);
        }

        return aggregated;
    }

    GraphHandle create(const GraphCreateInfo& graph_create_info) const {
        return GraphHandle(new Graph(graph_create_info));
    }
};

} // namespace merian
