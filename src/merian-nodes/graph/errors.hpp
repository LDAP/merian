#pragma once

#include <stdexcept>

namespace merian_nodes {

namespace graph_errors {

// general graph error
class graph_error : public std::runtime_error {
  public:
    graph_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

//---------------------

// error in a connector
class connector_error : public graph_error {
  public:
    connector_error(const std::string& what_arg) : graph_error(what_arg) {}
};

// error in a node
class node_error : public graph_error {
  public:
    node_error(const std::string& what_arg) : graph_error(what_arg) {}
};

//---------------------

// attempted to connect() a graph with an illegal connection present. For example, an input does not
// support the resource / output that is connected into it.)
class illegal_connection : public graph_error {
  public:
    illegal_connection(const std::string& what_arg) : graph_error(what_arg) {}
};

// attempted to connect() a graph with a missing connection. Meaning a node input was not connected
// to an node output.
class connection_missing : public graph_error {
  public:
    connection_missing(const std::string& what_arg) : graph_error(what_arg) {}
};

} // namespace graph_errors
} // namespace merian_nodes
