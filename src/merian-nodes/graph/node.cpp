#include "merian-nodes/graph/node.hpp"

namespace merian {

Node::NodeStatusFlags Node::load_config(const nlohmann::json& json) {
    merian::JSONLoadProperties props(json);
    std::ignore = properties(props);
    return {};
}

nlohmann::json Node::dump_config() {
    merian::JSONDumpProperties props;
    std::ignore = properties(props);
    return props.get();
}

} // namespace merian
