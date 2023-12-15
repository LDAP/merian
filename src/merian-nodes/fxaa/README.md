# FXAA

FXAA 3.11 by TIMOTHY LOTTES.

Inputs:

| Type  | Input ID | Input name | Description                                                               | Delay |
|-------|----------|------------|---------------------------------------------------------------------------|-------|
| Image | 0        | src        | the source image as RGBL, L in perceptual space (could be gamma 2.0)      | no    |

Outputs:

| Type  | Output ID | Output name | Description          | Format/Resolution          | Persistent |
|-------|-----------|-------------|----------------------|----------------------------|------------|
| Image | 0         | output      | antialiased image    | like `src`                 | no         |

