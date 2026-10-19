## Connectors

Connectors define the typed I/O of a node. The graph uses them to allocate resources, insert barriers, and wire descriptor set bindings automatically.

Names, accesses and delays are not part of the connector: they are supplied in
`InputConnectorDescriptor` / `OutputConnectorDescriptor` when returned from `describe_inputs()` /
`describe_outputs()`:

```cpp
{.name = "src", .connector = con_src, .access = ConnectorAccess::compute_read}
{.name = "out", .connector = con_out, .access = ConnectorAccess::compute_write}
```

### Delay

An input declared with `.delay = N` receives what its source wrote N iterations ago; delay 0
(default) means the current iteration's output of the upstream node. The output has a copy per
iteration of the graph's period (see Memory in the overview).

---

## Input connectors

| Connector | Factory | `io[input]` in `process()` |
|---|---|---|
| `VkSampledImageIn` | `create(overwrite_sampler = std::nullopt)` | `const ImageArrayResource&` (`get_image()`, `get_texture()`), bound as a combined image sampler |
| `VkStorageImageIn` | `create()` | `const ImageArrayResource&`, bound as a storage image |
| `VkBufferIn` | `create()` | `const BufferArrayResource&` (`get_buffer(index)`) |
| `PtrIn<T>` | `create()` | `const std::shared_ptr<T>&`; throws `node_error` if not connected (check `io.is_connected`) |
| `AnyIn` | `create()` | `const std::any&` |

Images stay in `eGeneral`. `PtrIn` / `PtrOut` and `AnyIn` / `AnyOut` pass host data through the
graph without GPU resources.

---

## Output connectors

### ManagedVkImageOut

The graph allocates the image; see the output declarations for its lifetime.

```cpp
ManagedVkImageOut::create(format, extent)
ManagedVkImageOut::create(format, width, height, depth = 1)
ManagedVkImageOut::create(create_infos)

// In process(): io[image_out] -> const ImageArrayResource&
```

### Output declarations

- `.persistent = true`: the contents survive iterations, so the node may write the output only when
  they change (an image loaded from disk, a solid color, an exposure average). It has one copy that
  no other resource shares and cannot be read delayed. Otherwise the contents do not survive the
  iteration: the output has a copy per iteration of the period (max delay + 1 with aliasing
  switched off), and a copy shares its memory with resources not in use while it is (from its write
  to the last read, delayed reads included).
  `Node::on_connected` can access only persistent outputs.
- `.scratch = true`: memory for the node itself (ping-pong targets, intermediate images), accessed
  and bound like an output but not connectable. A scratch output is in use only while its node runs,
  or survives iterations if also persistent.
- `.disabled = true`: no resource: `io[...]` throws `node_error`, image and buffer outputs bind a
  dummy and shader object outputs cannot be bound.

### ExternalVkImageOut

An image the node does not own (e.g. the Window node's swapchain image, `canvas`), set every
iteration. Accesses declared on it add no usage flags and it cannot be read delayed.

```cpp
ExternalVkImageOut::create()

// In process(): io[image_out].set(image_view) - nullptr when there is no image this iteration
// Consumers: io[image_in].get_image() / .get_texture() are null then.
```

### ManagedVkBufferOut

The graph allocates the buffer; see the output declarations for its lifetime.

```cpp
ManagedVkBufferOut::create(create_info, array_size = 1)

// In process(): io[buffer_out] -> const BufferArrayResource&
```

### PtrOut / AnyOut

```cpp
PtrOut<T>::create()
AnyOut::create()

// In process(): io[ptr_out] -> std::shared_ptr<T>&, io[any_out] -> std::any&; set every iteration
// unless the output is persistent
```

---

## Implementing connectors

- `InputConnector::can_receive_from(output)` decides whether the input can be connected to the
  output (the graph removes a user connection to the input it rejects); `on_connect_output(output)`
  then reads what the input needs from it, e.g. the array size.
- `OutputConnector::create_resource(const ResourceCreateInfo&)` is called once per copy and connect
  with the connected inputs, the combined access, the allocator for the copy, whether the output is
  persistent, the copy index and count and a debug name. Its memory is bound only before
  `OutputConnector::on_connected`. Memory shared with other resources needs no handling in the
  connector.
- `OutputConnector::on_connected(const ResourceConnectInfo&, Submission&, resource)` is called once
  per copy with the producing node's allocator: create views and descriptors and query device
  addresses here. What it records is submitted before the first run; a copy a delayed input reads
  before its producer first wrote it keeps what it writes.
- `OutputConnector::supports_delay()`: whether `create_resource` makes independent copies a delayed
  input can read.
- `GraphShaderObject::allocate(ShaderObjectAllocateInfo)` creates the images and buffers of a copy
  from the copy's allocator; `GraphShaderObject::on_connected(ShaderObjectConnectInfo, Submission&)`
  builds the shader objects with the producing node's allocator. `object()` is valid after it.
- `on_pre_process` / `on_post_process` add barriers to the supplied vectors; the graph records the
  barriers of all connectors of a layer at once, before and after the layer.
- Images stay in `eGeneral` between nodes; `process` may change the layout only of images the node
  writes.

---

## Binding

`io.bind(shader_object)` writes every shader-visible connector into the field of its name: input `src`
into `in_src`, output `out` into `out_out`, at the root or inside a `graph_in` / `graph_out` struct
field. Connectors without a field are skipped; unconnected optional inputs with a field receive a dummy.
