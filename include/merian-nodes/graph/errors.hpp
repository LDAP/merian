#pragma once

#include <stdexcept>

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

//---------------------

// Generic graph build error
class build_error : public graph_error {
  public:
    build_error(const std::string& what_arg) : graph_error(what_arg) {}
};

// attempted to build() a graph with an illegal connection present. For example, an input does not
// support the resource / output that is connected into it.)
class invalid_connection : public build_error {
  public:
    invalid_connection(const std::string& what_arg) : build_error(what_arg) {}
};

// attempted to build() a graph with a missing connection. Meaning a node input was not connected
// to an node output.
class connection_missing : public build_error {
  public:
    connection_missing(const std::string& what_arg) : build_error(what_arg) {}
};

} // namespace graph_errors
} // namespace merian
