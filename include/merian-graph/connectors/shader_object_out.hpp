#pragma once

#include "merian-graph/graph/connector_output.hpp"
#include "merian-graph/graph/errors.hpp"
#include "merian-graph/graph/graph_shader_object.hpp"
#include "merian-graph/graph/resource.hpp"

#include "merian/utils/pointer.hpp"

#include <memory>

namespace merian {

class ShaderObjectResource : public GraphResource {
  public:
    ShaderObjectResource(const GraphShaderObjectHandle& instance) : instance(instance) {}

    GraphShaderObjectHandle instance;
};

// Node-facing view of a transported GraphShaderObject.
// Implicitly converts to the shader object view matching the connector direction (inputs read,
// outputs and write-declared connectors read-write); use r()/w() explicitly where needed.
template <typename T> class ShaderObjectAccess {
  public:
    ShaderObjectAccess(const std::shared_ptr<T>& instance, const ShaderAccess access)
        : instance(instance), access(access) {}

    // Returned by value: the handles must outlive this view, which is a short-lived temporary
    // (`io[connector]`); the underlying object stays alive via the graph resource.
    T* operator->() const {
        return instance.get();
    }

    std::shared_ptr<T> get() const {
        return instance;
    }

    ShaderObjectHandle r() const {
        return instance->object(ShaderAccess::READ);
    }

    ShaderObjectHandle w() const {
        return instance->object(ShaderAccess::READ_WRITE);
    }

    operator ShaderObjectHandle() const {
        return instance->object(access);
    }

  private:
    std::shared_ptr<T> instance;
    ShaderAccess access;
};

template <typename T>
    requires std::derived_from<T, GraphShaderObject>
class ShaderObjectOut;
template <typename T> using ShaderObjectOutHandle = std::shared_ptr<ShaderObjectOut<T>>;

// Outputs a graph-allocated GraphShaderObject. One instance per copy: connect a delayed input to
// receive the instance of an earlier iteration.
template <typename T>
    requires std::derived_from<T, GraphShaderObject>
class ShaderObjectOut : public OutputConnector, public AccessibleConnector<ShaderObjectAccess<T>> {
  public:
    using CreateInfo = typename T::CreateInfo;

    ShaderObjectOut(const CreateInfo& create_info) : create_info(create_info) {}

    const CreateInfo& get_create_info() const {
        return create_info;
    }

    GraphResourceHandle create_resource(const ResourceCreateInfo& info) override {
        const std::shared_ptr<T> instance = create_instance(info.inputs);
        instance->allocate(ShaderObjectAllocateInfo{
            .context = info.allocator->get_context(),
            .allocator = info.allocator,
            .combined_access = info.combined_access,
            .persistent = info.persistent,
            .copy_index = info.copy_index,
            .copy_count = info.copy_count,
        });
        return std::make_shared<ShaderObjectResource>(instance);
    }

    ShaderObjectAccess<T> resource(const GraphResourceHandle& resource) override {
        return {std::static_pointer_cast<T>(
                    debugable_ptr_cast<ShaderObjectResource>(resource)->instance),
                ShaderAccess::READ_WRITE};
    }

    bool supports_delay() const override {
        return T::supports_delay;
    }

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              [[maybe_unused]] const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override {
        if (!resource) {
            throw graph_errors::node_error{"a disabled shader object output cannot be bound"};
        }
        cursor.write(debugable_ptr_cast<ShaderObjectResource>(resource)->instance->object(
            access.is_write() ? ShaderAccess::READ_WRITE : ShaderAccess::READ));
    }

    void on_connected(const ResourceConnectInfo& info,
                      Submission& submission,
                      const GraphResourceHandle& resource) override {
        const ContextHandle& context = info.allocator->get_context();
        debugable_ptr_cast<ShaderObjectResource>(resource)->instance->on_connected(
            ShaderObjectConnectInfo{
                .context = context,
                .compile_context = context->get_shader_compile_context(),
                .allocator = info.allocator,
            },
            submission);
    }

  public:
    static ShaderObjectOutHandle<T> create(const CreateInfo& create_info) {
        return std::make_shared<ShaderObjectOut<T>>(create_info);
    }

  protected:
    // The instance for one copy, before it is allocated.
    virtual std::shared_ptr<T> create_instance(
        [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs) {
        return std::make_shared<T>(create_info);
    }

  private:
    const CreateInfo create_info;
};

} // namespace merian
