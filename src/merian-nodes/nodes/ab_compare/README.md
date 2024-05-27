A simple A/B comparison nodes `ABSplitNode` and `ABSideBySideNode`. 

### `ABSplitNode`

Inputs:

| Type  | Input name | Description     | Delay |
|-------|------------|-----------------|-------|
| Image | a          | the left image  | no    |
| Image | b          | the right image | no    |

Outputs:

| Type  | Input ID | Input name | Description         | Format/Resolution        | Persistent |
|-------|----------|------------|---------------------|--------------------------|------------|
| Image | 0        | out        | split screen image  | user defined or like `a` | no         |


### `ABSideBySideNode`

Inputs:

| Type  | Input name | Description     | Delay |
|-------|------------|-----------------|-------|
| Image | a          | the left image  | no    |
| Image | b          | the right image | no    |

Outputs:

| Type  | Output name | Description         | Format/Resolution                    | Persistent |
|-------|-------------|---------------------|--------------------------------------|------------|
| Image | out         | split screen image  | user defined or twice as wide as `a` | no         |
