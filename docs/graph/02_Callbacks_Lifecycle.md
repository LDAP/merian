## Callbacks and Lifecycle

Note: topological order includes only connections with delay == 0.

### Graph::add_node

- `Node::describe_inputs` is called to learn which inputs the node requires.


### Graph Build

- All previous resources are released (calling their destructor). Outputs are re-read from nodes, allowing them to change.

- For each node (in topological order):
    - `Node::describe_outputs`
- For each node (in any order):
    - For each output:
        - For MAX_DELAY + 1 times:
            - `get_resource()`
            - `Resource::allocate`
- For each node (in any order):
    - Descriptor set layouts and descriptor sets are prepared.
- For each node (in topological order):
    - `Node::on_connected`
        - If `RESET_IN_FLIGHT_DATA` is set, all shared_ptrs for frame data are reset to nullptr.


### Graph Run

- Nodes are processed in topological order (excluding connections with delay > 0).

- For each node (in topological order):
    - `Node::pre_process`
    - If at least one node returns `NEEDS_RECONNECT`, a build is executed and the run starts over from the beginning.
- For each node (in topological order):
    - For each connector:
        - `Connector::on_pre_process`
    - Descriptor set writes are executed.
    - `Node::process`
    - For each connector:
        - `Connector::on_post_process`
