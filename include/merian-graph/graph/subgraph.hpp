#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace merian {

class Node;
using NodeHandle = std::shared_ptr<Node>;

// The children a node is made of, connected among each other and to the public ports of their
// parent.
class Subgraph {
  public:
    // The child's identifier is "<parent>:<name>".
    void add_node(const std::string& name, const NodeHandle& node);

    // Ports are "child.port", or ".port" for a public port of the parent: a public input as
    // source, a public output as destination.
    void add_connection(const std::string& src, const std::string& dst);

    // While the parent is disabled, its public output (".port") carries what its public input
    // receives.
    void add_pass_through(const std::string& input, const std::string& output);

    const std::vector<std::pair<std::string, NodeHandle>>& get_nodes() const;

    const std::vector<std::pair<std::string, std::string>>& get_connections() const;

    const std::vector<std::pair<std::string, std::string>>& get_pass_throughs() const;

  private:
    std::vector<std::pair<std::string, NodeHandle>> nodes;
    std::vector<std::pair<std::string, std::string>> connections;
    std::vector<std::pair<std::string, std::string>> pass_throughs;
};

} // namespace merian
