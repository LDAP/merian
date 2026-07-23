#pragma once

#include "connector_access.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

enum class ShaderAccess : uint8_t {
    READ,
    READ_WRITE,
};

struct ShaderObjectAllocateInfo {
    ContextHandle context;
    ShaderCompileContextHandle compile_context;
    ResourceAllocatorHandle allocator;
    ResourceAllocatorHandle aliasing_allocator;
    // Union of the declared ConnectorAccess over all connected ports - incorporate the usage
    // flags into allocations.
    ConnectorAccess combined_access;
    // 0 <= resource_index <= max_delay: which ring slot this instance is.
    uint32_t resource_index;
    uint32_t ring_size;
};

// A device object transported through the graph via ShaderObjectOut / ShaderObjectIn.
//
// The graph allocates one instance per ring slot (delayed inputs receive the previous
// iteration's instance). Allocation is free-form: device resources, host-side preparation,
// building and wiring the shader object(s).
class GraphShaderObject {
  public:
    virtual ~GraphShaderObject() = default;

    virtual void allocate(const ShaderObjectAllocateInfo& info) = 0;

    // The shader object to bind for the given access. Types without an access split return the
    // same object for both.
    virtual const ShaderObjectHandle& object(ShaderAccess access) const = 0;

    // Called once after (re)connect, before the first run. Initialize device resources here:
    // delayed inputs read a ring slot before its producer ever ran on it.
    virtual void on_connected([[maybe_unused]] const CommandBufferHandle& cmd) {}
};

using GraphShaderObjectHandle = std::shared_ptr<GraphShaderObject>;

} // namespace merian
