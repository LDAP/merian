# Image Sink

Writes to image files; writes run asynchronously, bounded by the graph's "max async tasks".

Inputs:

| Type  | Input ID | Input name | Description     | Delay |
|-------|----------|------------|-----------------|-------|
| Image | 0        | src        | the src image   | no    |

Events:

- `capture`: Sent in `process` if the current input was captured
- `start`: Sent in `pre_process` if recording started
- `stop`: Sent in `pre_process` if recording stopped
- `end`: Sent in `pre_process` when an exit trigger is reached
