# MedianApproxToBuffer

Calculates an approximation of the median and outputs it to a buffer.

Inputs:

| Type  | Input name | Description                                       | Delay |
|-------|------------|---------------------------------------------------|-------|
| Image | src        | input                                             | no    |

Outputs:

| Type   | Output name | Description             | Format/Resolution          | Persistent |
|--------|-------------|-------------------------|----------------------------|------------|
| Buffer | median      | approximation of median | float                      | no         |
| Buffer | histogram   | luminance histogram     | internal use only          | no         |
