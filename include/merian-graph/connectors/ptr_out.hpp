#pragma once

#include "merian-graph/resources/host_ptr_resource.hpp"

#include "merian-graph/graph/connector_output.hpp"
#include "merian-graph/graph/errors.hpp"
#include "merian-graph/graph/node.hpp"

#include <memory>

namespace merian {

template <typename T> class PtrOut;
template <typename T> using PtrOutHandle = std::shared_ptr<PtrOut<T>>;

// Transfer information between nodes on the host using shared_ptr.
template <typename T>
class PtrOut : public OutputConnector, public AccessibleConnector<std::shared_ptr<T>&> {

  public:
    PtrOut(const bool optional) : optional(optional) {}

    bool supports_in_place() const override {
        return true;
    }

    GraphResourceHandle create_resource(const ResourceCreateInfo& info) override {
        return std::make_shared<PtrResource<T>>(
            info.persistent ? -1 : static_cast<int32_t>(info.inputs.size()));
    }

    std::shared_ptr<T>& resource(const GraphResourceHandle& resource) override {
        return debugable_ptr_cast<PtrResource<T>>(resource)->ptr;
    }

    Connector::ConnectorStatusFlags on_post_process(
        [[maybe_unused]] Submission& submission,
        const GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override {
        const auto& res = debugable_ptr_cast<PtrResource<T>>(resource);
        if (!res->ptr && !optional) {
            throw graph_errors::connector_error{"Node did not set the resource for output."};
        }
        res->processed_inputs = 0;

        return {};
    }

  public:
    static PtrOutHandle<T> create() {
        return std::make_shared<PtrOut<T>>(false);
    }

    // An output a node may leave empty in an iteration. Every consumer has to handle the null
    // pointer, and the producer has to clear it: nothing resets the resource between iterations.
    static PtrOutHandle<T> create_optional() {
        return std::make_shared<PtrOut<T>>(true);
    }

  private:
    const bool optional;
};

} // namespace merian
