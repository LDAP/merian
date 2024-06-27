#include "any_in.hpp"
#include "any_out.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/resources/host_any_resource.hpp"

#include <memory>

namespace merian_nodes {

AnyIn::AnyIn(const std::string& name, const uint32_t delay)
    : TypedInputConnector<AnyOutHandle, const std::any&>(name, delay) {}

const std::any& AnyIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<AnyResource>(resource)->any;
}

void AnyIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<AnyOut>(output);
    if (!casted_output) {
        throw graph_errors::connector_error{
            fmt::format("AnyIn {} cannot recive from {}.", Connector::name, output->name)};
    }
}

Connector::ConnectorStatusFlags
AnyIn::on_post_process([[maybe_unused]] GraphRun& run,
                       [[maybe_unused]] const vk::CommandBuffer& cmd,
                       GraphResourceHandle& resource,
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

AnyInHandle AnyIn::create(const std::string& name, const uint32_t delay) {
    return std::make_shared<AnyIn>(name, delay);
}

} // namespace merian_nodes
