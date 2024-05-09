# Tone mapping

Tone mapping operator for high-dynamic range images.

Inputs:

| Type  | Input ID | Input name | Description                                       | Delay |
|-------|----------|------------|---------------------------------------------------|-------|
| Image | 0        | src        | the current frame                                 | no    |

Outputs:

| Type  | Output ID | Output name | Description         | Format/Resolution          | Persistent |
|-------|-----------|-------------|---------------------|----------------------------|------------|
| Image | 0         | output      | tonemapped image    | user-defined or like `src` | no         |

