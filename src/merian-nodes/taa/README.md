Temporal Anti-Aliasing (TAA) filter with magnitude dilation and different clamping modes.

Clamping modes:

- `MERIAN_NODES_TAA_CLAMP_NONE`
- `MERIAN_NODES_TAA_CLAMP_MIN_MAX`

Parameters:

| Name             | Type               | Description                                                         |
|------------------|--------------------|---------------------------------------------------------------------|
| `temporal_alpha` | float \[0, 1\]     | controls the blend with the previous frame. Higher means more reuse |


Inputs:

| Type  | Input ID | Input name | Description                                       | Delay |
|-------|----------|------------|---------------------------------------------------|-------|
| Image | 0        | current    | the current frame                                 | no    |
| Image | 1        | previous   | the previous frame (the previous output)          | 1     |
| Image | 2        | mv         | motion vectors in `r` and `g` channel in pixels   | no    |

Outputs:

| Type  | Input ID | Input name | Description         | Format/Resolution        | Persistent |
|-------|----------|------------|---------------------|--------------------------|------------|
| Image | 0        | output     | the blended image   | like `current`           | no         |


Example

```c++
auto taa = std::make_shared<merian::TAANode>(context, alloc, MERIAN_NODES_TAA_CLAMP_MIN_MAX);

graph.add_node("taa", taa);

graph.connect_image(taa, taa, 0, 1); // feedback
graph.connect_image(aliased_image, taa, 1, 0);
graph.connect_image(aliased_image_mv, taa, 0, 2);
```
