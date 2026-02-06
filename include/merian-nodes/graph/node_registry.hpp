#pragma once

#include "merian-nodes/graph/node.hpp"

#include <string>

namespace merian {

template <typename NodeClass>
NodeHandle create_node() {
    return std::make_shared<NodeClass>();
}

class NodeRegistry {

  public:
    using NodeFactory = std::function<NodeHandle()>;

    struct NodeTypeInfo {
        const std::string node_type_name;
        const std::string description;
        const NodeFactory factory;
    };

    struct NodeInfo {
        const std::type_index type;
        const std::string name;
        const std::string description;
        const std::optional<nlohmann::json> config = std::nullopt;
    };

  public:
    static NodeRegistry& get_instance();

    // Adds a new node type to this registy.
    //
    // If add_default_config is true a node with empty config is added.
    template <typename NODE_TYPE>
    void register_node_type(const std::string& node_type_name,
                            const std::string& description,
                            const bool add_default_config = true) {
        register_node_type<NODE_TYPE>(
            NodeTypeInfo{node_type_name, description, create_node<NODE_TYPE>}, add_default_config);
    }

    // Adds a new node type to this registy.
    //
    // If add_default_config is true a node with empty config is added.
    template <typename NODE_TYPE>
    void register_node_type(const NodeTypeInfo& node_info, const bool add_default_config = true) {
        const std::type_index type = typeid(std::remove_pointer_t<NODE_TYPE>);
        if (type_name_to_type.contains(node_info.node_type_name)) {
            throw std::invalid_argument{
                fmt::format("node with name {} already exists.", node_info.node_type_name)};
        }
        if (type_to_type_info.contains(type)) {
            throw std::invalid_argument{
                fmt::format("node with type {} already exists.", type.name())};
        }

        type_name_to_type.try_emplace(node_info.node_type_name, type);
        auto [it, _] = type_to_type_info.try_emplace(type, node_info);

        if (add_default_config) {
            try {
                register_node<NODE_TYPE>(it->second.node_type_name, it->second.description);
            } catch (const std::invalid_argument& e) {
                // revert...
                type_name_to_type.erase(it->second.node_type_name);
                type_to_type_info.erase(type);

                throw e;
            }
        }
    }

    // Adds a new node type to this registy.
    //
    // If add_default_config is true a node with empty config is added.
    template <typename NODE_TYPE>
    void register_node_type(NodeTypeInfo&& node_info, const bool add_default_config = true)
        requires(std::is_base_of_v<Node, NODE_TYPE>)
    {
        const std::type_index type = typeid(std::remove_pointer_t<NODE_TYPE>);
        if (type_name_to_type.contains(node_info.node_type_name)) {
            throw std::invalid_argument{
                fmt::format("node with type name {} already exists.", node_info.node_type_name)};
        }
        if (type_to_type_info.contains(type)) {
            throw std::invalid_argument{
                fmt::format("node with type {} already exists.", type.name())};
        }

        type_name_to_type.try_emplace(node_info.node_type_name, type);
        auto [it, _] = type_to_type_info.try_emplace(type, std::move(node_info));

        if (add_default_config) {
            try {
                register_node<NODE_TYPE>(it->second.node_type_name, it->second.description);
            } catch (const std::invalid_argument& e) {
                // revert...
                type_name_to_type.erase(it->second.node_type_name);
                type_to_type_info.erase(type);

                throw e;
            }
        }
    }

    template <typename NODE_TYPE>
    void register_node(const std::string& name,
                       const std::string& description,
                       const std::optional<nlohmann::json>& config = std::nullopt) {
        const std::string& type_name = node_type_name<NODE_TYPE>();
        register_node(type_name, name, description, config);
    }

    void register_node(const std::string& type_name,
                       const std::string& name,
                       const std::string& description,
                       const std::optional<nlohmann::json>& config = std::nullopt) {
        assert_node_type_exists(type_name);
        if (node_name_to_node_info.contains(name)) {
            throw std::invalid_argument{fmt::format("node with name {} already exists.", name)};
        }

        node_name_to_node_info.try_emplace(name, type_name_to_type.at(type_name), name, description,
                                           config);
        nodes.emplace_back(name);
        std::sort(nodes.begin(), nodes.end());
    }

    const std::vector<std::string>& node_names() const {
        return nodes;
    }

    auto node_type_names() const {
        return std::views::keys(type_name_to_type);
    }

    NodeHandle create_node_from_name(const std::string& name) {
        assert_node_name_exists(name);
        NodeInfo& node_info = node_name_to_node_info.at(name);
        NodeHandle node = type_to_type_info.at(node_info.type).factory();

        if (node_info.config) {
            node->load_config(*node_info.config);
        }

        return node;
    }

    NodeHandle create_node_from_type(const std::string& type_name,
                                     const std::optional<nlohmann::json>& config = std::nullopt) {
        assert_node_type_exists(type_name);
        const std::type_index& type = type_name_to_type.at(type_name);
        NodeHandle node = type_to_type_info.at(type).factory();

        if (config) {
            node->load_config(*config);
        }

        return node;
    }

    // shortcut for node_info(node).name;
    const std::string& node_type_name(const NodeHandle& node) const {
        return node_type_info(node).node_type_name;
    }

    const NodeTypeInfo& node_type_info(const NodeHandle& node) const {
        assert_node_type_exists(node);
        return type_to_type_info.at(typeindex_from_pointer(node));
    }

    const NodeTypeInfo& node_type_info(const std::string& node_type_name) const {
        assert_node_type_exists(node_type_name);
        return type_to_type_info.at(type_name_to_type.at(node_type_name));
    }

    const NodeInfo& node_info(const std::string& node_name) const {
        assert_node_name_exists(node_name);
        return node_name_to_node_info.at(node_name);
    }

    template <typename NODE_TYPE> const NodeTypeInfo& node_type_info() const {
        const std::type_index type = typeid(std::remove_pointer_t<NODE_TYPE>);
        if (!type_to_type_info.contains(type)) {
            throw std::invalid_argument{
                fmt::format("node with type {} was not registered.", type.name())};
        }
        return type_to_type_info.at(type);
    }

    // shortcut for node_info<TYPE>().node_type_name;
    template <typename NODE_TYPE> const std::string& node_type_name() const {
        return node_type_info<NODE_TYPE>().node_type_name;
    }

  private:
    NodeRegistry();

    void assert_node_name_exists(const std::string& node_name) const {
        if (!node_name_to_node_info.contains(node_name)) {
            throw std::invalid_argument{
                fmt::format("node with name {} was not registered.", node_name)};
        }
    }
    void assert_node_type_exists(const std::string& type_name) const {
        if (!type_name_to_type.contains(type_name)) {
            throw std::invalid_argument{
                fmt::format("node with type name {} was not registered.", type_name)};
        }
    }
    void assert_node_type_exists(const NodeHandle& node) const {
        if (!type_to_type_info.contains(typeindex_from_pointer(node))) {
            throw std::invalid_argument{fmt::format("node with type {} was not registered.",
                                                    typeindex_from_pointer(node).name())};
        }
    }

  private:
    std::vector<std::string> nodes;
    std::map<std::string, NodeInfo> node_name_to_node_info;

    std::map<std::string, std::type_index> type_name_to_type;
    std::map<std::type_index, NodeTypeInfo> type_to_type_info;
};

} // namespace merian
