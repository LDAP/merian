#pragma once

#include "ptr_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/resources/host_ptr_resource.hpp"

#include <memory>

namespace merian {

template <typename T> class PtrIn;
template <typename T> using PtrInHandle = std::shared_ptr<PtrIn<T>>;

// Transfer information between nodes on the host using shared_ptr.
template <typename T>
class PtrIn : public InputConnector,
              public OutputAccessibleInputConnector<PtrOutHandle<T>>,
              public AccessibleConnector<const std::shared_ptr<T>&> {

  public:
    PtrIn(const std::string& name, const uint32_t delay) : InputConnector(name, delay) {}

    void on_connect_output(const OutputConnectorHandle& output) override {
        auto casted_output = std::dynamic_pointer_cast<PtrOut<T>>(output);
        if (!casted_output) {
            throw graph_errors::invalid_connection{
                fmt::format("AnyIn {} cannot recive from {}.", Connector::name, output->name)};
        }
    }

    const std::shared_ptr<T>& resource(const GraphResourceHandle& resource) override {
        return debugable_ptr_cast<PtrResource<T>>(resource)->ptr;
    }

    Connector::ConnectorStatusFlags on_post_process(
        [[maybe_unused]] GraphRun& run,
        [[maybe_unused]] const CommandBufferHandle& cmd,
        const GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override {
        const auto& res = debugable_ptr_cast<PtrResource<T>>(resource);
        if ((++res->processed_inputs) == res->num_inputs) {
            // never happens if num_inputs == -1, which is used for persistent outputs.
            res->ptr.reset();
        }

        return {};
    }

  public:
    static PtrInHandle<T> create(const std::string& name, const uint32_t delay = 0) {
        return std::make_shared<PtrIn<T>>(name, delay);
    }
};

} // namespace merian
