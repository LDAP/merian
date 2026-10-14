## Graph

Merian provides a general purpose processing graph as well as some commonly used nodes.
Nodes define their inputs and outputs; the graph allocates the memory and calls the `pre_process` and `process` methods of nodes in topological order.
Nodes are grouped into dependency layers with one barrier between layers; the nodes of a layer may overlap on the GPU.

### Delayed inputs

Nodes can request a delayed view on an input: with delay `d` it receives what its source wrote `d`
iterations earlier; before the source wrote it, what its connector initialized (images: undefined
contents).

A delayed input can declare what it is connected to by default: `.auto_connect =
AutoConnect::output("out")` receives the node's own output, `AutoConnect::input("gbuffer")` what
another input receives, both with the input's delay. Unless the input is connected otherwise, the
graph makes the connection once; it is then an ordinary connection that can be removed or changed.
One to another input follows that input when it is rewired, or removed and connected again, unless
it was changed itself. One the graph removed because its source does not fit is made again once the
source changes; one the user removed stays removed.

### Pass-through

An input can declare `.pass_through = "out"`: while the user disabled the node, consumers of `out`
receive what the input receives (through chains of such nodes). The node shows as BYPASSED. Nodes
that are unsupported or erroneous but not disabled do not pass through, so the problem stays
visible.

### In-place inputs

An input declared with `.in_place = true` writes the resource it receives, with the declared access,
and passes it on as a new version: an output of the same name (e.g. `blit.canvas -> imgui.canvas`)
that the node does not declare. The other readers of a version run before its writer. Checked in
this order: a writer one of whose readers depends on it, or that reads the version through another
input, errors; if several in-place inputs read a version, all their writers error; a delayed reader
of a version that is written in place errors.
Whenever the node does not run (disabled, unsupported, erroneous, force disabled), the version
passes on unchanged, so a failing node does not take the UI drawn after it down; a node that fails
to describe its inputs passes nothing on. Image, buffer, ptr and any outputs can be written in
place.

### Subgraphs

A node can be made of other nodes. `describe_subgraph()` is called at the start of every connect and
returns its children, connected among each other (`"child.port"`) and to the node's public ports
(`".port"`):

```c++
// SVGF, shortened
Subgraph SVGF::describe_subgraph() {
    Subgraph subgraph;
    subgraph.add_node("variance estimate", variance_estimate);
    subgraph.add_connection(".src", "variance estimate.src");
    ...
    subgraph.add_node("taa", taa);
    subgraph.add_connection("filter 3.out", "taa.src");
    subgraph.add_connection("taa.out", ".out");
    subgraph.add_pass_through(".src", ".out");
    return subgraph;
}
```

- Children are registered node types (`register_node_type(..., false)` keeps them out of the add-node
  list) that the node constructs and keeps as members, so they survive reconnects; a variable number
  of children is created in the setter of the setting that shapes them. They run like other nodes as
  `<node>:<child>`; `:` is reserved in identifiers.
- Connections to a public port reach the children's ports; a public input feeds every child input
  connected to it. A child input connected to a public input that nothing wires keeps its own auto
  connection.
- Only the node is stored. It configures its children through setters and nests the settings of the
  children it persists in its own `properties()`, so they are stored under it and shown in its UI:

  ```c++
  if (config.st_begin_child("taa", "TAA")) {
      flags |= taa->properties(config);
      config.st_end_child();
  }
  ```

- Whenever the node does not run, its children do not run either; while the user disabled it, its
  `add_pass_through` ports pass through. The UI shows the children in the node's Subnodes tab.

### Memory

The graph decides where the memory of outputs comes from; connectors create their resources from
the allocator they are given and need no handling of shared memory.

- The period is lcm(delay + 1) over the delayed inputs of the nodes that run. An output has a copy
  per iteration of the period, copy `c` written in the iterations `i ≡ c (mod period)`. Persistent
  outputs and outputs whose connector does not support delay have one copy; with "alias transient
  resources" switched off an output has max delay + 1 copies.
- A copy is in use from its write to its last read (delayed reads included), a scratch output while
  its node runs, a persistent one always. Copies whose times of use do not overlap share memory:
  they are placed greedy by size (Pisarchyk & Lee 2020) into blocks of the memory type the standard
  allocator would choose, as many as needed; a reconnect reuses the blocks that fit. What cannot be
  placed (host-mapped memory, raw allocations, ...) is allocated by the node's allocator.
- A non-persistent output starts each iteration with undefined contents: the node writes it before
  reading it. The graph transitions an image from `UNDEFINED` when another resource used its memory
  since its last use, and orders the hand-over of memory with a barrier on all commands.
- Each node receives its own allocator in `initialize` (also `NodeIO::get_allocator()`), which counts
  what it allocated.

`Graph::get_memory_report()` (printed by `merian-graph-run --print-memory`) lists the bytes in use at
each layer of the period, the blocks with the placement and time of use of every copy and, for each
node in run order, what it allocated and the copies of its outputs. The graph's Memory tab shows it
with a plot of the bytes in use; a node's Memory tab shows its part.

### Example

```c++
    auto graph = std::make_shared<merian::Graph>(merian::GraphCreateInfo{context, alloc});

    auto window_node = std::make_shared<merian::GLFWWindowNode>(context);
    auto image_in    = std::make_shared<merian::HDRImageRead>(alloc->get_staging(), "filename.hdr");
    auto exp         = std::make_shared<merian::AutoExposure>(context);
    auto tonemap     = std::make_shared<merian::Tonemap>(context);
    auto curve       = std::make_shared<merian::VKDTFilmcurv>(context);

    graph->add_node(window_node, "window");
    graph->add_node(image_in,    "image_in");
    graph->add_node(exp,         "exp");
    graph->add_node(tonemap,     "tonemap");
    graph->add_node(curve,       "curve");

    graph->add_connection("image_in", "exp",     "out", "src");
    graph->add_connection("exp",      "tonemap",  "out", "src");
    graph->add_connection("tonemap",  "curve",    "out", "src");
    graph->add_connection("curve",    "window",   "out", "src");

    while (!window_node->get_window()->should_close()) {
        glfwPollEvents();
        graph->run();
    }
    graph->wait();
```

### Available nodes

| Class | Purpose |
|---|---|
| `GLFWWindowNode` | Swapchain output window |
| `HDRImageRead` | Load HDR/LDR image from disk |
| `ImageWrite` | Write image to disk |
| `AutoExposure` | Auto exposure / histogram |
| `Tonemap` | Tone mapping operators |
| `VKDTFilmcurv` | Film curve (from vkdt) |
| `Bloom` | Bloom post-process |
| `FXAA` | Fast approximate anti-aliasing |
| `TAA` | Temporal anti-aliasing |
| `SVGF` | Spatiotemporal variance-guided filtering |
| `DLSSNode` | NVIDIA DLSS super resolution and ray reconstruction |
| `Accumulate` | Temporal accumulation |
| `DeviceASBuilder` | Acceleration structure builder |
| `GBufferRTNode` | G-buffer via ray tracing |
| `ColorImage` | Solid color image source |
| `AbstractCompute` | Generic compute pass base class |
| `AbstractABCompare` | A/B image comparison |
| `Reduce` | Parallel reduction |
| `MeanToBuffer` | Image mean reduction |
| `MedianApproxNode` | Approximate median |
| `Shadertoy` | Shadertoy-style compute node |
