#include "merian-graph/connectors/any_in.hpp"
#include "merian-graph/connectors/any_out.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian-graph/resources/host_any_resource.hpp"

#include <memory>

namespace merian {

const std::any& AnyIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<AnyResource>(resource)->any;
}

void AnyIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<AnyOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{"AnyIn cannot receive from output."};
    }
}

Connector::ConnectorStatusFlags
AnyIn::on_post_process([[maybe_unused]] GraphRun& run,
                       [[maybe_unused]] const CommandBufferHandle& cmd,
                       const GraphResourceHandle& resource,
                       [[maybe_unused]] const NodeHandle& node,
                       [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                       [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    const auto& res = debugable_ptr_cast<AnyResource>(resource);
    if ((++res->processed_inputs) == res->num_inputs) {
        // never happens if num_inputs == -1, which is used for persistent outputs.
        res->any.reset();
    }

    return {};
}

AnyInHandle AnyIn::create() {
    return std::make_shared<AnyIn>();
}

} // namespace merian
