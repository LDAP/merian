#include "merian-graph/connectors/any_out.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian-graph/resources/host_any_resource.hpp"

namespace merian {

GraphResourceHandle AnyOut::create_resource(const ResourceCreateInfo& info) {
    return std::make_shared<AnyResource>(
        info.persistent ? -1 : static_cast<int32_t>(info.inputs.size()));
}

std::any& AnyOut::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<AnyResource>(resource)->any;
}

Connector::ConnectorStatusFlags
AnyOut::on_pre_process([[maybe_unused]] Submission& submission,
                       const GraphResourceHandle& resource,
                       [[maybe_unused]] const NodeHandle& node,
                       [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                       [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    const auto& res = debugable_ptr_cast<AnyResource>(resource);
    if (res->num_inputs != -1) {
        res->any.reset();
    }

    return {};
}

Connector::ConnectorStatusFlags
AnyOut::on_post_process([[maybe_unused]] Submission& submission,
                        const GraphResourceHandle& resource,
                        [[maybe_unused]] const NodeHandle& node,
                        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    const auto& res = debugable_ptr_cast<AnyResource>(resource);
    if (!res->any.has_value()) {
        throw graph_errors::connector_error{"Node did not set the resource for output."};
    }
    res->processed_inputs = 0;

    return {};
}

AnyOutHandle AnyOut::create() {
    return std::make_shared<AnyOut>();
}

} // namespace merian
