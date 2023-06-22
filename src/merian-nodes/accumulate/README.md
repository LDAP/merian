Accumulate (sum) the float RGBA values of an input image (src, 0) to a persistent output image (dst, 0).

Inputs:

| Type  | Input ID | Input name | Description     | Delay |
|-------|----------|------------|-----------------|-------|
| Image | 0        | src        | the src image   | no    |

Outputs:

| Type  | Input ID | Input name | Description         | Format/Resolution        | Persistent |
|-------|----------|------------|---------------------|--------------------------|------------|
| Image | 0        | dst        | the sum of src      | like `src`               | yes        |
