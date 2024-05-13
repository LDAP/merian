#pragma once

#include <stdexcept>

namespace merian_nodes {

namespace graph_errors {

class unsupported_connection : public std::runtime_error {
  public:
    unsupported_connection(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

class connector_error : public std::runtime_error {
  public:
    connector_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

} // namespace graph_errors
} // namespace merian_nodes
