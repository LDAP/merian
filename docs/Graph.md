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


