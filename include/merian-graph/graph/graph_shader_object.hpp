#pragma once

#include "connector_access.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/vk/command/submission.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

enum class ShaderAccess : uint8_t {
    READ,
    READ_WRITE,
};

struct ShaderObjectAllocateInfo {
    ContextHandle context;
    // Memory for this copy, not bound before on_connected.
    ResourceAllocatorHandle allocator;
    // Union of the declared ConnectorAccess over all connected connectors - incorporate the usage
    // flags into allocations.
    ConnectorAccess combined_access;
    // The producer relies on the contents surviving iterations.
    bool persistent;
    // Copy c is written in the iterations i with i % copy_count == c.
    uint32_t copy_index;
    uint32_t copy_count;
};

struct ShaderObjectConnectInfo {
    ContextHandle context;
    ShaderCompileContextHandle compile_context;
    // Of the producing node, for the shader objects, views and uploads.
    ResourceAllocatorHandle allocator;
};

// A device object transported through the graph via ShaderObjectOut / ShaderObjectIn.
//
// The graph creates one instance per copy.
class GraphShaderObject {
  public:
    // The instances are independent: a delayed input can read the one of an earlier iteration.
    static constexpr bool supports_delay = true;

    virtual ~GraphShaderObject() = default;

    // Create the images and buffers of this copy.
    virtual void allocate([[maybe_unused]] const ShaderObjectAllocateInfo& info) {}

    // Called once after (re)connect, before the first run: build the shader objects here and
    // initialize the resources a delayed input may read before the producer wrote them.
    virtual void on_connected(const ShaderObjectConnectInfo& info, Submission& submission) = 0;

    // The shader object to bind for the given access, valid after on_connected. Types without an
    // access split return the same object for both.
    virtual const ShaderObjectHandle& object(ShaderAccess access) const = 0;
};

using GraphShaderObjectHandle = std::shared_ptr<GraphShaderObject>;

} // namespace merian
