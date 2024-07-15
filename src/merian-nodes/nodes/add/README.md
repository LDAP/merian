### `AddNode`

Adds the two input images together.

Inputs:

| Type      | Input name | Delay |
|-----------|------------|-------|
| VkImageIn | image_0    | no    |
| VkImageIn | image_1    | no    |
| VkImageIn | ...        | no    |

Outputs:

| Type       | Input name | Description         | Format/Resolution                                                               | Persistent |
|------------|------------|---------------------|---------------------------------------------------------------------------------|------------|
| VkImageOut | out        | sum of inputs       | format: user defined, equal to `a`, equal to `a`, resolution: min of all images | no         |
