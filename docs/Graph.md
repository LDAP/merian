## Graph

Merian provides a general purpose processing graph as well as some commonly used nodes (as merian-nodes).
Nodes define their input and outputs, the graph allocates the memory and calls the `cmd_build` and `cmd_process` methods of nodes.
Before `cmd_process` can issue commands (e.g. rebuild/skip/...) to the graph by implementing the `pre_process` method.

### Delayed inputs

Nodes can request a delayed view on an input.
The graph internally creates (delay + 1) resources and cycles them in every iteration.

### Persistent outputs

Nodes can request a persistent output. Persistent outputs are guaranteed to contain the same content between runs.
Currently, it is not possible to have a delayed view on a persistent output (because the data would have to be copied after each iteration).
You can accomplish this by using a "copy" node in between.
Persistent outputs are not persistent between graph builds.

### Rules

- Nodes can access the same resource (same output with same delay) with the same layout only (we do not duplicate the resource for you).


### Example:

```c++
merian::Graph graph{context, alloc, queue};
auto output = std::make_shared<merian::GLFWWindowNode<merian::FIT>>(context, window, surface, queue);
auto ab = std::make_shared<merian::ABCompareNode>();
auto image_node = std::make_shared<merian::ImageNode>(alloc, "/home/lucas/Downloads/image.jpg", loader, false);
auto spheres = std::make_shared<merian::ShadertoySpheresNode>(context, alloc);

graph.add_node("output", output);
graph.add_node("a", spheres);
graph.add_node("b", image_node);
graph.add_node("ab compare", ab);
graph.connect_image(spheres, ab, 0, 0);
graph.connect_image(image_node, ab, 0, 1);
graph.connect_image(ab, output, 0, 0);

// Optionally, you can build the graph here, since building is a expensive operation.
// This makes the first run faster.
// However many nodes need frequent rebuilds, for example when the Window resolution changes.

auto ring_cmd_pool = make_shared<merian::RingCommandPool<>>(context, context->queue_family_idx_GCT);
auto ring_fences = make_shared<merian::RingFences<>>(context);
while (!glfwWindowShouldClose(*window)) {
    auto frame_data = ring_fences->wait_and_get();
    auto cmd_pool = ring_cmd_pool->set_cycle();
    glfwPollEvents();

    auto cmd = cmd_pool->create_and_begin();
    auto run = graph.cmd_run(cmd);
    cmd_pool->end_all();

    queue->submit(cmd_pool, frame_data.fence, run.get_signal_semaphore(),
                  run.get_wait_semaphores(), run.get_wait_stages());
    run.execute_callbacks(queue);
}
```
