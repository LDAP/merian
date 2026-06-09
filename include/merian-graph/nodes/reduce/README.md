### `ReduceInputs`

Reduce the multiple images together.

Inputs:

| Type      | Input name | Delay |
|-----------|------------|-------|
| VkImageIn | image_0    | no    |
| VkImageIn | image_1    | no    |
| VkImageIn | ...        | no    |

Outputs:

| Type       | Input name | Description         | Format/Resolution                                                               | Persistent |
|------------|------------|---------------------|---------------------------------------------------------------------------------|------------|
| VkImageOut | out        | reduced inputs      | format: user defined, or equal to `a`, resolution: min of all images            | no         |
