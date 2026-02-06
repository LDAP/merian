#pragma once

#include "merian-nodes/graph/node_registry.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/utils/vector.hpp"

namespace merian {

class ExtensionGraphNodes : public ContextExtension {
  public:
    ExtensionGraphNodes() : ContextExtension("ExtensionGraphNodes") {}

    std::vector<std::string> request_extensions() override {
        std::vector<std::string> aggregated;

        auto& registry = NodeRegistry::get_instance();
        for (const auto& type_name : registry.node_type_names()) {
            auto node = registry.create_node_from_type(type_name);
            insert_all(aggregated, node->request_context_extensions());
        }

        return aggregated;
    }

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override {
        InstanceSupportInfo aggregated;
        aggregated.supported = true;

        auto& registry = NodeRegistry::get_instance();
        for (const auto& type_name : registry.node_type_names()) {
            auto node = registry.create_node_from_type(type_name);
            auto support_info = node->query_instance_support(query_info);

            insert_all(aggregated.required_extensions, support_info.required_extensions);
            insert_all(aggregated.required_layers, support_info.required_layers);
        }

        return aggregated;
    }

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override {
        DeviceSupportInfo aggregated;
        aggregated.supported = true;

        auto& registry = NodeRegistry::get_instance();
        for (const auto& type_name : registry.node_type_names()) {
            auto node = registry.create_node_from_type(type_name);
            auto support_info = node->query_device_support(query_info);

            insert_all(aggregated.required_extensions, support_info.required_extensions);
            insert_all(aggregated.required_features, support_info.required_features);
            insert_all(aggregated.required_spirv_extensions, support_info.required_spirv_extensions);
            insert_all(aggregated.required_spirv_capabilities, support_info.required_spirv_capabilities);
        }

        return aggregated;
    }
};

} // namespace merian
