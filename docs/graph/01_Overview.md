## Graph

Merian provides a general purpose processing graph as well as some commonly used nodes.
Nodes define their input and outputs, the graph allocates the memory and calls the `pre_process` and `process` methods of nodes.

### Delayed inputs

Nodes can request a delayed view on an input.
The graph internally creates (delay + 1) resources and cycles them in every iteration.

### Example:

```c++
    merian::Graph graph{context, alloc};
    auto window_node = std::make_shared<merian::GLFWWindow>(context);
    auto image_in = std::make_shared<merian::HDRImageRead>(alloc->getStaging(), "filename.hdr", false);
    auto exp = std::make_shared<merian::AutoExposure>(context);
    auto tonemap = std::make_shared<merian::Tonemap>(context);
    auto curve = std::make_shared<merian::VKDTFilmcurv>(context);

    graph.add_node(window_node, "window");
    graph.add_node(image_in);
    graph.add_node(exp);
    graph.add_node(tonemap);
    graph.add_node(curve);

    graph.add_connection(image_in, exp, "out", "src");
    graph.add_connection(exp, tonemap, "out", "src");
    graph.add_connection(tonemap, curve, "out", "src");
    graph.add_connection(curve, window_node, "out", "src");

    while (!window_node->get_window()->should_close()) {
        glfwPollEvents();
        graph.run();
    }
```
