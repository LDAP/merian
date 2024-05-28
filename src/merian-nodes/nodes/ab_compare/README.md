Simple A/B comparison nodes `ABSplitNode` and `ABSideBySideNode`. 

### `ABSplitNode`

Inputs:

| Type      | Input name | Description     | Delay |
|-----------|------------|-----------------|-------|
| VkImageIn | a          | the left image  | no    |
| VkImageIn | b          | the right image | no    |

Outputs:

| Type       | Input name | Description         | Format/Resolution        | Persistent |
|------------|------------|---------------------|--------------------------|------------|
| VkImageOut | out        | split screen image  | user defined or like `a` | no         |


### `ABSideBySideNode`

Inputs:

| Type      | Input name | Description     | Delay |
|-----------|------------|-----------------|-------|
| VkImageIn | a          | the left image  | no    |
| VkImageIn | b          | the right image | no    |

Outputs:

| Type       | Output name | Description         | Format/Resolution                    | Persistent |
|------------|-------------|---------------------|--------------------------------------|------------|
| VkImageOut | out         | split screen image  | user defined or twice as wide as `a` | no         |
