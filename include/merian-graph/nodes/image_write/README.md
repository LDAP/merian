# Image Write

Writes to image files. Note that this is slow since it does wait for idle after every iteration and blocks while writing the file.

Inputs:

| Type  | Input ID | Input name | Description     | Delay |
|-------|----------|------------|-----------------|-------|
| Image | 0        | src        | the src image   | no    |

Events:

- `capture`: Sent in `process` if the current input was captured
- `start`: Sent in `pre_process` if recording started
- `stop`: Sent in `pre_process` if recording stopped
