## Graph

Merian provides a general purpose processing graph as well as some commonly used nodes (as merian-nodes).
Nodes define their input and outputs, the graph allocates the memory and calls the `cmd_build` and `cmd_process` methods of nodes.
Before `cmd_process` can issue commands (e.g. rebuild/skip/...) to the graph by implementing the `pre_process` method.



