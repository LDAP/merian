#pragma once

#include "merian-nodes/resources/host_ptr_resource.hpp"

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

#include <memory>

namespace merian {

template <typename T> class PtrOut;
template <typename T> using PtrOutHandle = std::shared_ptr<PtrOut<T>>;

// Transfer information between nodes on the host using shared_ptr.
template <typename T>
class PtrOut : public OutputConnector, public AccessibleConnector<std::shared_ptr<T>&> {

  public:
    PtrOut(const std::string& name, const bool persistent)
        : OutputConnector(name, !persistent), persistent(persistent) {}

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
                    [[maybe_unused]] const uint32_t resource_index,
                    [[maybe_unused]] const uint32_t ring_size) override {
        return std::make_shared<PtrResource<T>>(persistent ? -1 : (int32_t)inputs.size());
    }

    std::shared_ptr<T>& resource(const GraphResourceHandle& resource) override {
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
        if (!res->ptr) {
            throw graph_errors::connector_error{
                fmt::format("Node did not set the resource for output {}.", Connector::name)};
        }
        res->processed_inputs = 0;

        return {};
    }

  public:
    static PtrOutHandle<T> create(const std::string& name, const bool persistent = false) {
        return std::make_shared<PtrOut<T>>(name, persistent);
    }

  private:
    const bool persistent;
};

} // namespace merian
