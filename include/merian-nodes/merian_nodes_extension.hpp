#pragma once

#include "merian-nodes/graph/graph.hpp"
#include "merian-nodes/graph/node_registry.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/extension_glsl_compiler.hpp"

namespace merian {

class MerianNodesExtension : public ContextExtension {
  public:
    MerianNodesExtension() : ContextExtension() {}

    std::vector<std::string> request_extensions() override {
        std::vector<std::string> aggregated;

        // Request compiler extensions for nodes that need shader compilation
        aggregated.push_back("merian-glsl-compiler");

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
        for (const auto& type : registry.node_types()) {
            auto it = instance_support_cache.find(type);
            if (it == instance_support_cache.end()) {
                const auto node = registry.create_node_from_type(type);
                std::tie(it, std::ignore) =
                    instance_support_cache.emplace(type, query_instance_support(query_info, node));
                SPDLOG_DEBUG("node {} instance support: {}", registry.node_type_name(type),
                             it->second);
            }

            insert_all(aggregated.required_extensions, it->second.required_extensions);
            insert_all(aggregated.required_layers, it->second.required_layers);
        }

        return aggregated;
    }

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override {
        DeviceSupportInfo aggregated = DeviceSupportInfo::check(query_info, {
                                                                                "timelineSemaphore",
                                                                                "hostQueryReset",
                                                                            });

        if (!aggregated.supported) {
            return aggregated;
        }

        if (query_info.extension_container.get_context_extension<ExtensionGLSLCompiler>(true) ==
            nullptr) {
            return DeviceSupportInfo{false, "merian-glsl-compiler must be supported."};
        }

        auto& registry = NodeRegistry::get_instance();
        for (const auto& type : registry.node_types()) {
            auto& current_device_support_cache =
                all_device_support_cache[query_info.physical_device];

            auto it = current_device_support_cache.find(type);
            if (it == current_device_support_cache.end()) {
                const auto node = registry.create_node_from_type(type);
                std::tie(it, std::ignore) =
                    all_device_support_cache[query_info.physical_device].emplace(
                        type, query_device_support(query_info, node));
                SPDLOG_DEBUG("node {} device support: {}", registry.node_type_name(type),
                             it->second);
            }

            insert_all(aggregated.required_extensions, it->second.required_extensions);
            insert_all(aggregated.required_features, it->second.required_features);
            insert_all(aggregated.required_spirv_extensions, it->second.required_spirv_extensions);
            insert_all(aggregated.required_spirv_capabilities,
                       it->second.required_spirv_capabilities);
        }

        return aggregated;
    }

    /* Called after the physical device was select and before extensions are checked for
     * compatibility and check_support is called.*/
    void on_physical_device_selected(const PhysicalDeviceHandle& physical_device,
                                     const ExtensionContainer& /*extension_container*/) override {
        device_support_cache = std::move(all_device_support_cache.at(physical_device));
        all_device_support_cache.clear();
    }

    // ---------------------------------

    GraphHandle create(const GraphCreateInfo& graph_create_info) const {
        return GraphHandle(new Graph(graph_create_info));
    }

    const InstanceSupportInfo& get_instance_support(const NodeHandle& node) {
        const auto& registry = NodeRegistry::get_instance();

        const auto it = instance_support_cache.find(registry.node_type(node));
        assert(it != instance_support_cache.end() &&
               "nodes types must be registered before context creation");
        return it->second;
    }

    const DeviceSupportInfo& get_device_support(const NodeHandle& node) {
        const auto& registry = NodeRegistry::get_instance();

        const auto it = device_support_cache.find(registry.node_type(node));
        assert(it != device_support_cache.end() &&
               "nodes types must be registered before context creation");
        return it->second;
    }

  private:
    static DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info,
                                                  const NodeHandle& node) {
        DeviceSupportInfo support_info;
        try {
            support_info = node->query_device_support(query_info);
        } catch (const ShaderCompiler::compilation_failed& e) {
            support_info =
                DeviceSupportInfo{false, fmt::format("compilation failed: {}", e.what())};
        }
        return support_info;
    }

    static InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info,
                                                      const NodeHandle& node) {
        InstanceSupportInfo support_info;
        try {
            support_info = node->query_instance_support(query_info);
        } catch (const ShaderCompiler::compilation_failed& e) {
            support_info =
                InstanceSupportInfo{false, fmt::format("compilation failed: {}", e.what())};
        }
        return support_info;
    }

  private:
    std::unordered_map<std::type_index, InstanceSupportInfo> instance_support_cache;
    std::unordered_map<PhysicalDeviceHandle, std::unordered_map<std::type_index, DeviceSupportInfo>>
        all_device_support_cache;
    std::unordered_map<std::type_index, DeviceSupportInfo> device_support_cache;
};

} // namespace merian
