## Callbacks and Lifecycle

### Graph::add_node

- Node::describe_inputs


### Graph Build

- All previous resources are released (calling their destructor). Outputs are read again from nodes, allowing them to change.

- Nodes are processed in topological order (excluding connections with delay > 0).

- for each node:
    - Node::describe_outputs
    - for each output:
        - for MAX_DELAY + 1 times
            - get_resource()
            - Resource::allocate
- for each node:
    - Descriptor set layouts and descriptor sets are prepared
    - Node::on_connected
        - If RESET_IN_FLIGHT_DATA is set, then all shared_ptrs for frame data are reset to nullptr.

### Garph Run

- Nodes are processed in topological order (excluding connections with delay > 0).

- for each node:
    - Node::pre_process
    - For each connector:
        - Connector::on_pre_process
    - for each resource corresponding to an output connector:
        - GraphResource::get_status (descriptor set writes are prepared)
    - If for at least on node NEEDS_REBUILD is set, then a build is executed and the run begins again from the start.
- for each node:
    - (descriptor set writes are executed)
    - Node::process
    - For each connector:
        - Connector::on_pre_process
