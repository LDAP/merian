## Graph

Merian provides a general purpose processing graph as well as some commonly used nodes.
Nodes define their inputs and outputs; the graph allocates the memory and calls the `pre_process` and `process` methods of nodes in topological order.

### Delayed inputs

Nodes can request a delayed view on an input.
The graph internally creates (delay + 1) resources and cycles them every iteration.

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
