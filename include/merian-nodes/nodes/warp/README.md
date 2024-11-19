## Warp

Warps an image according to pixel offsets.

The node is configurable to input and output multiple images and buffers.
For buffers the index computation can be selected as formula or by choosing from a set of defaults.

Inputs:

| Type              | Input name   | Description                              | Delay |
|-------------------|--------------|------------------------------------------|-------|
| VkImageIn         | mv           | motion vectors in `r` and `g` channel    | no    |
| VkImage/BufferIn  | src1         | input image or buffer                    | no    |
| VkImage/BufferIn  | src2         | input image or buffer                    | no    |
| VkImage/BufferIn  | ...          | input image or buffer                    | no    |

Outputs:

| Type              | Output name   | Description                      | Format/Resolution | Persistent |
|-------------------|---------------|----------------------------------|-------------------|------------|
| VkImage/BufferOut | out1          | the warped input image or buffer | like input        | no         |
| VkImage/BufferOut | out2          | the warped input image or buffer | like input        | no         |
| VkImage/BufferOut | ...           | the warped input image or buffer | like input        | no         |
