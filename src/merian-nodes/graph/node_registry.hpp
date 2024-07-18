#pragma once

#include "merian-nodes/graph/node.hpp"

#include <string>

namespace merian_nodes {

class NodeRegistry {

  public:
    struct NodeInfo {
        const std::string name;
        const std::string description;
        const std::function<NodeHandle()> factory;
    };

  public:
    NodeRegistry(const ContextHandle& context, const ResourceAllocatorHandle& allocator);

    template <typename NODE_TYPE> void register_node(const NodeInfo& node_info) {
        const std::type_index type = typeid(std::remove_pointer_t<NODE_TYPE>);
        if (name_to_type.contains(node_info.name)) {
            throw std::invalid_argument{
                fmt::format("node with name {} already exists.", node_info.name)};
        }
        if (type_to_info.contains(type)) {
            throw std::invalid_argument{
                fmt::format("node with type {} already exists.", type.name())};
        }

        nodes.emplace_back(node_info.name);
        name_to_type.try_emplace(node_info.name, type);
        type_to_info.try_emplace(type, node_info);
        std::sort(nodes.begin(), nodes.end());
    }

    template <typename NODE_TYPE> void register_node(const NodeInfo&& node_info) {
        const std::type_index type = typeid(std::remove_pointer_t<NODE_TYPE>);
        if (name_to_type.contains(node_info.name)) {
            throw std::invalid_argument{
                fmt::format("node with name {} already exists.", node_info.name)};
        }
        if (type_to_info.contains(type)) {
            throw std::invalid_argument{
                fmt::format("node with type {} already exists.", type.name())};
        }

        nodes.emplace_back(node_info.name);
        name_to_type.try_emplace(node_info.name, type);
        type_to_info.try_emplace(type, std::move(node_info));
        std::sort(nodes.begin(), nodes.end());
    }

    const std::vector<std::string>& node_names() const {
        return nodes;
    }

    NodeHandle create_node_from_name(const std::string& name) {
        assert_node_exists(name);
        return node_info(name).factory();
    }

    // shortcut for node_info(node).name;
    const std::string& node_name(const NodeHandle& node) const {
        return node_info(node).name;
    }

    const NodeInfo& node_info(const NodeHandle& node) const {
        assert_node_exists(node);
        return type_to_info.at(typeindex_from_pointer(node));
    }

    const NodeInfo& node_info(const std::string& name) const {
        assert_node_exists(name);
        return type_to_info.at(name_to_type.at(name));
    }

    template <typename NODE_TYPE> const NodeInfo& node_info() const {
        const std::type_index type = typeid(std::remove_pointer_t<NODE_TYPE>);
        if (!type_to_info.contains(type)) {
            throw std::invalid_argument{
                fmt::format("node with type {} was not registered.", type.name())};
        }
        return type_to_info.at(type);
    }

    // shortcut for node_info<TYPE>().name;
    template <typename NODE_TYPE> const std::string& node_name() const {
        return node_info<NODE_TYPE>().name;
    }

  private:
    void assert_node_exists(const std::string& name) const {
        if (!name_to_type.contains(name)) {
            throw std::invalid_argument{fmt::format("node with name {} was not registered.", name)};
        }
    }
    void assert_node_exists(const NodeHandle& node) const {
        if (!type_to_info.contains(typeindex_from_pointer(node))) {
            throw std::invalid_argument{fmt::format("node with type {} was not registered.",
                                                    typeindex_from_pointer(node).name())};
        }
    }

  private:
    std::vector<std::string> nodes;
    std::map<std::string, std::type_index> name_to_type;
    std::map<std::type_index, NodeInfo> type_to_info;
};

} // namespace merian_nodes
