#include "merian-graph/graph/errors.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace merian::graph_errors {

config_error::config_error(const std::vector<std::string>& messages)
    : graph_error(fmt::format("{}", fmt::join(messages, "\n"))) {}

} // namespace merian::graph_errors
