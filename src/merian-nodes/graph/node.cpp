#include "merian-nodes/graph/node.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian/utils/properties_json_dump.hpp"
#include "merian/utils/properties_json_load.hpp"

namespace merian {

void Node::initialize(const ContextHandle& context, const ResourceAllocatorHandle& /*allocator*/) {
    const DeviceSupportQueryInfo query_info{
        context->get_file_loader(), context->get_physical_device(), context->get_queue_info(),
        *context, context->get_shader_compile_context()};
    const DeviceSupportInfo support_info = query_device_support(query_info);

    if (support_info.supported) {
        return;
    }

    throw graph_errors::node_error{support_info.unsupported_reason.empty()
                                       ? "device does not support this node"
                                       : support_info.unsupported_reason};
}

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
