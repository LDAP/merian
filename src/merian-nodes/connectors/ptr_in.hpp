#pragma once

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/resources/host_ptr_resource.hpp"

#include <memory>

namespace merian_nodes {

template <typename T> class PtrIn;
template <typename T> using PtrInHandle = std::shared_ptr<PtrIn<T>>;
template <typename T> class PtrOut;
template <typename T> using PtrOutHandle = std::shared_ptr<PtrOut<T>>;

// Transfer information between nodes on the host using shared_ptr.
template <typename T>
class PtrIn : public TypedInputConnector<PtrOutHandle<T>, const std::shared_ptr<T>&> {

  public:
    PtrIn(const std::string& name, const uint32_t delay)
        : TypedInputConnector<PtrOutHandle<T>, const std::shared_ptr<T>&>(name, delay) {}

    const std::shared_ptr<T>& resource(const GraphResourceHandle& resource) override {
        return debugable_ptr_cast<PtrResource<T>>(resource)->ptr;
    }

    Connector::ConnectorStatusFlags on_post_process(
        [[maybe_unused]] GraphRun& run,
        [[maybe_unused]] const vk::CommandBuffer& cmd,
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

} // namespace merian_nodes
