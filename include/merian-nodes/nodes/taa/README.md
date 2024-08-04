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
| Image | feedback   | the previous frame (the previous output)          | 1     |
| Image | mv         | motion vectors in `r` and `g` channel in pixels   | no    |

#### Outputs:

| Type  | Output name | Description         | Format/Resolution        | Persistent |
|-------|-------------|---------------------|--------------------------|------------|
| Image | out         | the blended image   | like `current`           | no         |


#### Example

```c++
auto taa = std::make_shared<merian::TAANode>(context, alloc, MERIAN_NODES_TAA_CLAMP_MIN_MAX);

graph.add_node("taa", taa);

graph.connect_image(taa, taa, "output", "previous"); // feedback
graph.connect_image(aliased_image, taa, "out", "current");
graph.connect_image(aliased_image_mv, taa, "out_mv", "mv");
```
