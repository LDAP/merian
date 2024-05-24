## Graph

Merian provides a general purpose processing graph as well as some commonly used nodes.
Nodes define their input and outputs, the graph allocates the memory and calls the `pre_process` and `process` methods of nodes.

### Delayed inputs

Nodes can request a delayed view on an input.
The graph internally creates (delay + 1) resources and cycles them in every iteration.

### Example:

```c++
merian::Graph graph{context, alloc, queue};
auto output = std::make_shared<merian::GLFWWindowNode>(context, queue);
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

while (!output->get_window()->should_close()) {
    glfwPollEvents();
    graph.run();
}
```
