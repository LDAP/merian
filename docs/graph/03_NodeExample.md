## Node Implementation Example

A compute node that blends its input with its own output of the previous iteration.

```cpp
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node.hpp"

#include "merian/shader/slang_entry_point.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

class Blend : public merian::Node {
  public:
    // Called on every connect.
    std::vector<merian::InputConnectorDescriptor> describe_inputs() override {
        return {
            {.name = "src", .connector = con_src, .access = merian::ConnectorAccess::compute_read},
            // the output of the previous iteration, unless the user connects it otherwise
            {.name = "prev_out",
             .connector = con_prev_out,
             .access = merian::ConnectorAccess::compute_read,
             .delay = 1,
             .auto_connect = merian::AutoConnect::output("out")},
        };
    }

    // Called on every connect, once the sources of the undelayed inputs described their outputs.
    std::vector<merian::OutputConnectorDescriptor>
    describe_outputs(const merian::NodeIOLayout& io_layout) override {
        const vk::ImageCreateInfo src = io_layout[con_src]->get_create_info_or_throw();
        con_out = merian::ManagedVkImageOut::create(src.format, src.extent);
        return {{.name = "out",
                 .connector = con_out,
                 .access = merian::ConnectorAccess::compute_write}};
    }

    // Called after every connect, when no work is in flight: build pipelines here. Only persistent
    // outputs can be accessed.
    NodeStatusFlags on_connected([[maybe_unused]] const merian::NodeIOLayout& io_layout,
                                 const merian::NodeIO& io,
                                 [[maybe_unused]] const merian::NodeConnectionInfo& info,
                                 [[maybe_unused]] merian::Submission& submission) override {
        const merian::ContextHandle& context = io.get_context();
        entry_point = merian::SlangProgramEntryPoint::create(
                          merian::SlangProgram::create(context->get_shader_compile_context(),
                                                       "blend.slang"),
                          "main")
                          .get();
        auto spec_builder = merian::SpecializationInfoBuilder();
        spec_builder.add_entry(local_size, local_size);
        pipeline = merian::ComputePipeline::create(entry_point->get_pipeline_layout(context),
                                                   entry_point->specialize(spec_builder.build()));
        globals = entry_point->create_global_shader_object(context, io.get_allocator());
        return {};
    }

    // Records the GPU work. `out` is not persistent: `prev_out` is undefined until it was written.
    NodeStatusFlags process(const merian::NodeIO& io,
                            const merian::NodeProcessInfo& info,
                            merian::Submission& submission) override {
        const merian::CommandBufferHandle& cmd = submission.get_cmd();
        // in_src, in_prev_out and out_out in the shader
        io.bind(globals);
        cmd->bind(pipeline);
        entry_point->bind_global(globals, cmd, pipeline, info.get_shader_object_allocator());
        pc.reset = static_cast<VkBool32>(info.get_iteration() == 0);
        cmd->push_constant(pipeline, pc);
        cmd->dispatch(con_out->get_create_info()->extent, local_size, local_size);
        return {};
    }

    // Settings and status; stored with the graph. Return NEEDS_RECONNECT for changes that need
    // new outputs or pipelines.
    NodeStatusFlags properties(merian::Properties& config) override {
        config.config_float("alpha", pc.alpha, "More means more reuse.", 0.01f, 0.0f, 1.0f);
        return {};
    }

  private:
    struct PushConstant {
        float alpha = 0.9f;
        VkBool32 reset;
    };

    static constexpr uint32_t local_size = 16;
    PushConstant pc;

    const merian::VkSampledImageInHandle con_src = merian::VkSampledImageIn::create();
    const merian::VkSampledImageInHandle con_prev_out = merian::VkSampledImageIn::create();
    merian::ManagedVkImageOutHandle con_out;

    merian::SlangProgramEntryPointHandle entry_point;
    merian::PipelineHandle pipeline;
    merian::ShaderObjectHandle globals;
};
```

`blend.slang`:

```slang
[vk::constant_id(0)]
const int local_size_x = 1;
[vk::constant_id(1)]
const int local_size_y = 1;

Sampler2D<float4> in_src;
Sampler2D<float4> in_prev_out;
RWTexture2D<float4> out_out;

struct PushConstant {
    float alpha;
    int reset;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

[numthreads(local_size_x, local_size_y, 1)]
[shader("compute")]
void main(const uint2 pixel: SV_DispatchThreadID) {
    uint2 size;
    out_out.GetDimensions(size.x, size.y);
    if (any(pixel >= size))
        return;
    const float4 src = in_src[pixel];
    const float4 previous = params.reset != 0 ? src : in_prev_out[pixel];
    out_out[pixel] = lerp(src, previous, params.alpha);
}
```

### Status flags

| Flag | Effect |
|---|---|
| `NEEDS_RECONNECT` | Graph rebuilds before the next `process()` call. |
| `RESET_IN_FLIGHT_DATA` | All per-frame shared_ptrs are reset to nullptr. |
| `REMOVE_NODE` | Node is removed from the graph after the current run. |

### NodeIO quick reference

```cpp
io[input_connector]                  // Access input resource (type from AccessibleConnector<T>)
io[output_connector]                 // Access output resource
io.is_connected(connector)           // Check whether connector is wired up
io.frame_data<T>(ctor_args...)       // Per-in-flight-slot persistent storage
io.bind(shader_object)               // Writes the connectors into the fields in_<input> / out_<output>
io.send_event("name", data)          // Broadcast an inter-node event
io.get_context()                     // The context
io.get_allocator()                   // The node's memory, the allocator initialize received
io.get_seed()                        // The node's seed, from the graph seed and the identifier
```

### NodeIOLayout quick reference (available in describe_outputs / on_connected)

```cpp
io[input_connector]                           // Upstream OutputConnector
io.is_connected(input_connector)              // Is the input wired?
io.get_context(), io.get_allocator()          // As in NodeIO
io.register_event_listener("//", callback)    // Listen to events ("//" = any)
// Pattern format: "NodeType/identifier/eventName"  (empty field = wildcard), comma-separated
```

Send a fixed event name and let the receiver pick what listens to it. For a node that offers
several actions, hold a `NodeEvents` and declare them with their handlers:

```cpp
NodeEvents events{{
    {"play", [this] { playing = true; }},
    {"pause", [this] { playing = false; }},
}};
```

`events.register_listeners(io_layout)` in `describe_outputs` registers the patterns and
`events.properties(config)` draws them; a changed pattern needs a reconnect.
