## Tone mapping

Tone mapping operator for high-dynamic range images.

### Inputs:

| Type  | Input name | Description                                       | Delay |
|-------|------------|---------------------------------------------------|-------|
| Image | src        | the current frame                                 | no    |

### Outputs:

| Type  | Output name | Description         | Format/Resolution          | Persistent |
|-------|-------------|---------------------|----------------------------|------------|
| Image | out         | tonemapped image    | user-defined or like `src` | no         |

