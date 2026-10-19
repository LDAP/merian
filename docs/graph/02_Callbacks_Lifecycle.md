## Callbacks and Lifecycle

Note: topological order includes only connections with delay == 0.

### Graph::add_node

- `Node::initialize`; a node that fails to initialize is unsupported.

### Graph::remove_node, Graph::clear, ~Graph

- Once the iterations in flight finished: `Node::on_removed`, for the children of a subgraph too.
  Others may still hold the node.


### Graph Build

- All previous resources are released (calling their destructor). Outputs are re-read from nodes, allowing them to change.

- For each supported node, before its children: `Node::describe_subgraph`; children are added,
  removed, moved and renamed accordingly.
- For each node whose subgraph could be described (also unsupported ones): `Node::describe_inputs`
- The auto connections of the inputs outside subgraphs that were never connected are made (they may
  wire a public input).
- Each subgraph is checked, parents before their children; the child inputs it does not connect,
  or connects to a public input nothing wires, get their auto connection.
- For each node (in identifier order, once the sources of its undelayed inputs described their outputs):
    - Its undelayed inputs are decided: connected, unconnected, stopped (the source does not run or
      disabled the output) or rejected (`InputConnector::can_receive_from`, delay support, self
      loops, missing and scratch outputs). Connections from disabled nodes are followed to what they
      pass through, versions whose writer does not run to what that writer received.
    - `Node::describe_outputs`; each in-place input adds an output of its name.
- Delayed inputs and inputs whose source stopped later are decided; nodes stop until nothing changes.
- Rejected user connections that reach the input directly are removed and the build restarts; at nodes
  that do not run, connections that cannot work for any input are removed as well.
- For each connected input: `InputConnector::on_connect_output`
- In-place writes are validated; a rejection errors the node and restarts the build. Rejections and errors from `on_connected` / `pre_process` / `process` hold until a build raises none and requests no other.
- Nodes are grouped into dependency layers; the readers of a version come before its in-place writer. The period is lcm(delay + 1) over the delayed inputs.
- For each output that is not disabled or written in place (a chain of versions shares the resources of its root):
    - For each copy (one if the output is persistent or does not support delay, else one per iteration of the period, or max delay + 1 with aliasing switched off): `OutputConnector::create_resource`
- All copies are placed into shared memory where their lifetimes over the period do not overlap, then bound.
- For each copy: `OutputConnector::on_connected` (views are created here); last the copies a delayed input reads before their producer first wrote them.
- For each node (in topological order):
    - `Node::on_connected`
        - If `RESET_IN_FLIGHT_DATA` is set, all shared_ptrs for frame data are reset to nullptr.


### Graph Run

- While waiting for the slot of the iteration in flight it reuses, the host callbacks of
  `Submission::sync_to_cpu` whose commands finished run, oldest iteration first. A run ends by
  running the finished ones; `Graph::wait()` runs all.
- For each node (in topological order):
    - `Node::pre_process`
    - If at least one node returns `NEEDS_RECONNECT`, a build is executed and the run starts over from the beginning.
- For each layer:
    - For each connector of its nodes, except outputs written in place: `Connector::on_pre_process`
    - One barrier: the layer's memory barrier, the discards of the images whose memory another resource used since their last use and the barriers of the connectors.
    - For each node: `Node::process`. The nodes of a layer may overlap on the GPU (not while the
      profiler is enabled). `process` may change the layout only of images the node writes.
    - For each connector of its nodes, except outputs written in place: `Connector::on_post_process`
    - One barrier with the barriers of the connectors.
- The last batch signals the iteration semaphore with the total iteration + 1. When recording
  throws, what was recorded is still submitted and the exception propagates.
