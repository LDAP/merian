A simple A/B comparison node.

The node has two inputs `a` and `b`.
If not specified, the output image will have the format and dimensions of `a`.

Inputs:

| Type  | Input ID | Input name | Description     | Delay |
|-------|----------|------------|-----------------|-------|
| Image | 0        | a          | the left image  | no    |
| Image | 1        | b          | the right image | no    |

Outputs:

| Type  | Input ID | Input name | Description         | Format/Resolution        | Persistent |
|-------|----------|------------|---------------------|--------------------------|------------|
| Image | 0        | result     | split screen image  | user defined or like `a` | no         |

