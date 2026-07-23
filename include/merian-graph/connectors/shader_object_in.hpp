#pragma once

#include "shader_object_out.hpp"

#include "merian-graph/graph/connector_input.hpp"
#include "merian-graph/graph/errors.hpp"

#include <memory>

namespace merian {

template <typename T>
    requires std::derived_from<T, GraphShaderObject>
class ShaderObjectIn;
template <typename T> using ShaderObjectInHandle = std::shared_ptr<ShaderObjectIn<T>>;

// Receives a GraphShaderObject. With delay > 0 the previous iteration's instance is delivered.
template <typename T>
    requires std::derived_from<T, GraphShaderObject>
class ShaderObjectIn : public InputConnector,
                       public OutputAccessibleInputConnector<ShaderObjectOutHandle<T>>,
                       public AccessibleConnector<ShaderObjectAccess<T>> {
  public:
    ShaderObjectIn() = default;

    void on_connect_output(const OutputConnectorHandle& output) override {
        if (!std::dynamic_pointer_cast<ShaderObjectOut<T>>(output)) {
            throw graph_errors::invalid_connection{
                "ShaderObjectIn expects a ShaderObjectOut of the same type."};
        }
    }

    ShaderObjectAccess<T> resource(const GraphResourceHandle& resource) override {
        return {std::static_pointer_cast<T>(
                    debugable_ptr_cast<ShaderObjectResource>(resource)->instance),
                ShaderAccess::READ};
    }

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              [[maybe_unused]] const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override {
        if (!resource) {
            return; // unconnected optional input: no dummy exists for arbitrary object types
        }
        cursor.write(debugable_ptr_cast<ShaderObjectResource>(resource)->instance->object(
            access.is_write() ? ShaderAccess::READ_WRITE : ShaderAccess::READ));
    }

  public:
    static ShaderObjectInHandle<T> create() {
        return std::make_shared<ShaderObjectIn<T>>();
    }
};

} // namespace merian
