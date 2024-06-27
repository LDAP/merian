## Callbacks and Lifecycle

Note, topological order includes all connections with delay == 0.

### Graph::add_node

- Node::describe_inputs


### Graph Build

- All previous resources are released (calling their destructor). Outputs are read again from nodes, allowing them to change.

- for each node (in topological order):
    - Node::describe_outputs
- for each node (in any order):
    - for each output:
        - for MAX_DELAY + 1 times
            - get_resource()
            - Resource::allocate
- for each node (in any order):
    - Descriptor set layouts and descriptor sets are prepared
- for each node (in topological order):
    - Node::on_connected
        - If RESET_IN_FLIGHT_DATA is set, then all shared_ptrs for frame data are reset to nullptr.

### Graph Run

- Nodes are processed in topological order (excluding connections with delay > 0).

- for each node (in topological order):
    - Node::pre_process
    - If for at least on node NEEDS_REBUILD is set, then a build is executed and the run begins again from the start.
- for each node (in topological order):
    - For each connector:
        - Connector::on_pre_process
    - (descriptor set writes are executed)
    - Node::process
    - For each connector:
        - Connector::on_pre_process
