### `AddNode`

Adds the two input images together.

Inputs:

| Type      | Input name | Delay |
|-----------|------------|-------|
| VkImageIn | a          | no    |
| VkImageIn | b          | no    |

Outputs:

| Type       | Input name | Description         | Format/Resolution                                  | Persistent |
|------------|------------|---------------------|----------------------------------------------------|------------|
| VkImageOut | out        | sum of a and b      | priority: user defined, equal to `a`, equal to `a` | no         |
