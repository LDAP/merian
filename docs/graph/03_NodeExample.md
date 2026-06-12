## Node Implementation Example

A complete example of a custom compute node that reads a sampled image and writes a storage image.

```cpp
#include "merian-graph/graph/node.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"  // VkSampledImageIn
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"

class MyProcessNode : public merian::Node {
  public:
    // Connector handles declared as members so they remain accessible in all lifecycle callbacks.
    merian::VkSampledImageInHandle input;
    merian::ManagedVkImageOutHandle output;

    // Optional: request context extensions before Context creation.
    std::vector<std::string> request_context_extensions() override {
        return {};
    }

    // Declare inputs. Called once when the node is added to the graph.
    std::vector<merian::InputConnectorDescriptor> describe_inputs() override {
        input = merian::VkSampledImageIn::compute_read();
        return {{"input", input}};
    }

    // Declare outputs. Called each reconnect, after all inputs are resolved.
    // io[input] gives the upstream output connector, from which format/extent can be queried.
    std::vector<merian::OutputConnectorDescriptor>
    describe_outputs(const merian::NodeIOLayout& io) override {
        // Mirror the upstream image format and extent.
        auto upstream = io[input];
        vk::Format format = upstream->create_info.format;
        vk::Extent3D extent = upstream->create_info.extent;

        output = merian::ManagedVkImageOut::compute_write(format, extent);
        return {{"output", output}};
    }

    // Called after the graph is fully connected. Build pipelines here.
    merian::NodeStatusFlags
    on_connected(const merian::NodeIOLayout& io,
                 const merian::DescriptorSetLayoutHandle& layout) override {
        // layout contains bindings for all connectors in declaration order.
        pipeline_layout = merian::PipelineLayoutBuilder(context)
            .add_descriptor_set_layout(layout)
            .add_push_constant<MyPushConstants>()
            .build_pipeline_layout();

        auto session = merian::SlangSession::create(
            merian::ShaderCompileContext::create(context));
        auto entry_point = session->load_module_from_path_and_compile_entry_point(
            context, "my_shader.slang");
        pipeline = merian::ComputePipeline::create(pipeline_layout, entry_point);

        return {};
    }

    // CPU-side work before GPU dispatch. Return NEEDS_RECONNECT to trigger a graph rebuild.
    merian::NodeStatusFlags pre_process(const merian::GraphRun& run,
                                        const merian::NodeIO& io) override {
        // Example: force reconnect if a parameter changed that affects output size.
        if (output_size_changed) {
            return Node::NEEDS_RECONNECT;
        }
        return {};
    }

    // GPU work. Record commands into run.get_cmd().
    void process(merian::GraphRun& run,
                 const merian::DescriptorSetHandle& desc,
                 const merian::NodeIO& io) override {
        auto& cmd = run.get_cmd();

        // Access resources via operator[].
        const auto& in_res  = io[input];   // BufferArrayResource or ImageArrayResource
        const auto& out_res = io[output];

        // Per-frame persistent data (constructed once per in-flight slot).
        auto& frame = io.frame_data<MyFrameData>();

        // Push constants and dispatch.
        MyPushConstants pc{run.get_iteration(), run.get_time_delta()};
        cmd->push_constant(pipeline, pc);
        cmd->bind(pipeline);
        cmd->bind_descriptor_set(pipeline, 0, desc);
        cmd->dispatch(out_res.image->get_extent(), 8, 8);
    }

  private:
    merian::PipelineLayoutHandle pipeline_layout;
    merian::PipelineHandle pipeline;
    merian::ContextHandle context;
    bool output_size_changed = false;

    struct MyPushConstants { uint32_t iteration; float time_delta; };
    struct MyFrameData { /* per-in-flight-slot data */ };
};
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
io.get_binding(input_connector)      // Descriptor set binding index for input
io.get_binding(output_connector)     // Descriptor set binding index for output
io.send_event("name", data)          // Broadcast an inter-node event
```

### NodeIOLayout quick reference (available in describe_outputs / on_connected)

```cpp
io[input_connector]                           // Upstream OutputConnector
io.is_connected(input_connector)              // Is the input wired?
io.register_event_listener("", callback)      // Listen to events (empty = any)
// Pattern format: "NodeType/identifier/eventName"  (empty field = wildcard)
```
