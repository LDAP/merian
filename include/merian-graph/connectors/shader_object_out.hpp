#pragma once

#include "merian-graph/graph/connector_output.hpp"
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
// Implicitly converts to the shader object view matching the port direction (inputs read,
// outputs and write-declared ports read-write); use r()/w() explicitly where needed.
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

// Outputs a graph-allocated GraphShaderObject. One instance per ring slot: connect a delayed
// input to receive the previous iteration's instance.
template <typename T>
    requires std::derived_from<T, GraphShaderObject>
class ShaderObjectOut : public OutputConnector, public AccessibleConnector<ShaderObjectAccess<T>> {
  public:
    using Factory = std::function<std::shared_ptr<T>()>;

    ShaderObjectOut(const Factory& factory, const bool persistent)
        : OutputConnector(!persistent), factory(factory), persistent(persistent) {}

    GraphResourceHandle create_resource(
        [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
        const ConnectorAccess& combined_access,
        const ResourceAllocatorHandle& allocator,
        const ResourceAllocatorHandle& aliasing_allocator,
        const uint32_t resource_index,
        const uint32_t ring_size) override {
        const std::shared_ptr<T> instance = factory();
        const ContextHandle& context = allocator->get_context();
        instance->allocate(ShaderObjectAllocateInfo{
            .context = context,
            .compile_context = context->get_shader_compile_context(),
            .allocator = allocator,
            .aliasing_allocator = aliasing_allocator,
            .combined_access = combined_access,
            .resource_index = resource_index,
            .ring_size = ring_size,
        });
        return std::make_shared<ShaderObjectResource>(instance);
    }

    ShaderObjectAccess<T> resource(const GraphResourceHandle& resource) override {
        return {std::static_pointer_cast<T>(
                    debugable_ptr_cast<ShaderObjectResource>(resource)->instance),
                ShaderAccess::READ_WRITE};
    }

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              [[maybe_unused]] const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override {
        cursor.write(debugable_ptr_cast<ShaderObjectResource>(resource)->instance->object(
            access.is_write() ? ShaderAccess::READ_WRITE : ShaderAccess::READ));
    }

    void on_connected(const CommandBufferHandle& cmd,
                      const std::vector<GraphResourceHandle>& resources) override {
        for (const GraphResourceHandle& resource : resources) {
            debugable_ptr_cast<ShaderObjectResource>(resource)->instance->on_connected(cmd);
        }
    }

  public:
    static ShaderObjectOutHandle<T> create(const Factory& factory, const bool persistent = false) {
        return std::make_shared<ShaderObjectOut<T>>(factory, persistent);
    }

  private:
    const Factory factory;

  public:
    const bool persistent;
};

} // namespace merian
