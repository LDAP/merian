#include "merian-graph/graph/subgraph.hpp"

namespace merian {

void Subgraph::add_node(const std::string& name, const NodeHandle& node) {
    nodes.emplace_back(name, node);
}

void Subgraph::add_connection(const std::string& src, const std::string& dst) {
    connections.emplace_back(src, dst);
}

void Subgraph::add_pass_through(const std::string& input, const std::string& output) {
    pass_throughs.emplace_back(input, output);
}

const std::vector<std::pair<std::string, NodeHandle>>& Subgraph::get_nodes() const {
    return nodes;
}

const std::vector<std::pair<std::string, std::string>>& Subgraph::get_connections() const {
    return connections;
}

const std::vector<std::pair<std::string, std::string>>& Subgraph::get_pass_throughs() const {
    return pass_throughs;
}

} // namespace merian
