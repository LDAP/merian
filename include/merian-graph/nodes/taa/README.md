### Temporal Anti-Aliasing (TAA)

Filter with magnitude dilation and different clamping modes.

#### Clamping modes:

- `MERIAN_NODES_TAA_CLAMP_NONE`
- `MERIAN_NODES_TAA_CLAMP_MIN_MAX`

#### Parameters:

| Name             | Type               | Description                                                         |
|------------------|--------------------|---------------------------------------------------------------------|
| `alpha`          | float \[0, 1\]     | controls the blend with the previous frame. Higher means more reuse |


#### Inputs:

| Type  | Input name | Description                                       | Delay |
|-------|------------|---------------------------------------------------|-------|
| Image | src        | the current frame                                 | no    |
| Image | prev_src   | the previous output, connected by default         | 1     |
| Image | mv         | motion vectors in `r` and `g` channel in pixels   | no    |

#### Outputs:

| Type  | Output name | Description         | Format/Resolution        | Persistent |
|-------|-------------|---------------------|--------------------------|------------|
| Image | out         | the blended image   | like `src`, or the overwrite format | no |


#### Example

```c++
graph->add_node("TAA", "taa");
graph->add_connection("render", "taa", "out", "src");
graph->add_connection("render", "taa", "mv", "mv");
```
