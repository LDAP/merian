#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace merian {

namespace graph_errors {

// Generic graph error
class graph_error : public std::runtime_error {
  public:
    graph_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

//---------------------

// error in a connector - might be handled by the graph builder or runtime
class connector_error : public graph_error {
  public:
    connector_error(const std::string& what_arg) : graph_error(what_arg) {}
};

// error in a node - might be handled by the graph builder or runtime
class node_error : public graph_error {
  public:
    node_error(const std::string& what_arg) : graph_error(what_arg) {}
};

// what() lists all messages, one per line.
class config_error : public graph_error {
  public:
    config_error(const std::vector<std::string>& messages);
};

} // namespace graph_errors
} // namespace merian
