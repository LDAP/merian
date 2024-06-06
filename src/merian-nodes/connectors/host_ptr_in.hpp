#pragma once

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/resources/host_ptr_resource.hpp"

#include <memory>

namespace merian_nodes {

template <typename T> class HostPtrIn;
template <typename T> using HostPtrInHandle = std::shared_ptr<HostPtrIn<T>>;
template <typename T> class HostPtrOut;
template <typename T> using HostPtrOutHandle = std::shared_ptr<HostPtrOut<T>>;

// Transfer information between nodes on the host using shared_ptr.
template <typename T>
class HostPtrIn : public TypedInputConnector<HostPtrOut<T>, const std::shared_ptr<T>&> {

  public:
    HostPtrIn(const std::string& name, const uint32_t delay)
        : TypedInputConnector<HostPtrOut<T>, const std::shared_ptr<T>&>(name, delay) {}

    const std::shared_ptr<T>& resource(const GraphResourceHandle& resource) override {
        return debugable_ptr_cast<HostPtrResource<T>>(resource)->ptr;
    }

    Connector::ConnectorStatusFlags on_post_process(
        [[maybe_unused]] GraphRun& run,
        [[maybe_unused]] const vk::CommandBuffer& cmd,
        GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override {
        const auto& res = debugable_ptr_cast<HostPtrResource<T>>(resource);
        if ((++res->processed_inputs) == res->num_inputs) {
            // never happens if num_inputs == -1, which is used for persistent outputs.
            res->ptr.reset();
        }

        return {};
    }

  public:
    static HostPtrInHandle<T> create(const std::string& name, const uint32_t delay = 0) {
        return std::make_shared<HostPtrIn<T>>(name, delay);
    }
};

} // namespace merian_nodes
