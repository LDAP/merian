#include "merian-nodes/graph/node.hpp"

#include "merian/utils/properties_json_dump.hpp"
#include "merian/utils/properties_json_load.hpp"

namespace merian {

void Node::initialize(const ContextHandle& /*context*/,
                      const ResourceAllocatorHandle& /*allocator*/) {}

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
