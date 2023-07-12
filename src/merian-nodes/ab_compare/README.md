A simple A/B comparison nodes `ABSplitNode` and `ABSideBySideNode`. 

### `ABSplitNode`

Inputs:

| Type  | Input ID | Input name | Description     | Delay |
|-------|----------|------------|-----------------|-------|
| Image | 0        | a          | the left image  | no    |
| Image | 1        | b          | the right image | no    |

Outputs:

| Type  | Input ID | Input name | Description         | Format/Resolution        | Persistent |
|-------|----------|------------|---------------------|--------------------------|------------|
| Image | 0        | result     | split screen image  | user defined or like `a` | no         |


### `ABSideBySideNode`

Inputs:

| Type  | Input ID | Input name | Description     | Delay |
|-------|----------|------------|-----------------|-------|
| Image | 0        | a          | the left image  | no    |
| Image | 1        | b          | the right image | no    |

Outputs:

| Type  | Output ID | Output name | Description         | Format/Resolution                    | Persistent |
|-------|-----------|-------------|---------------------|--------------------------------------|------------|
| Image | 0         | result      | split screen image  | user defined or twice as wide as `a` | no         |
