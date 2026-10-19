#include "merian-graph/graph/node.hpp"

#include "merian/utils/properties_json_dump.hpp"
#include "merian/utils/properties_json_load.hpp"

namespace merian {

void Node::initialize(const ContextHandle& /*context*/,
                      const ResourceAllocatorHandle& /*allocator*/) {}

Node::NodeStatusFlags Node::load_config(const nlohmann::json& json,
                                        std::vector<std::string>& problems) {
    NodeStatusFlags flags = 0;
    problems = JSONLoadProperties::load_revealed(json, [&](Properties& props) {
        flags |= properties(props);
        // a node that leaves reads no more
        return (flags & REMOVE_NODE) == 0u;
    });
    return flags;
}

nlohmann::json Node::dump_config() {
    merian::JSONDumpProperties props;
    std::ignore = properties(props);
    return props.get();
}

} // namespace merian
