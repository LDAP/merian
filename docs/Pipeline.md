## Descriptors

Merian provides a builder for `PipelineLayout` as well as several abstractions.
The idea is that you do not need to remember things that you did when creating the objects, the objects remember these things for you.
Also, you do not need to keep pointers alive if not explicitly stated.

- `SpecializationInfoBuilder`: A builder for `SpecializationInfo`.
- `SpecializationInfo`: Wraps a specialization info and stores the values internally.
- `PipelineLayoutBuilder`: A builder for `PipelineLayout`
- `PipelineLayout`: Wraps a pipeline layout and destroys it in its destructor.
- `Pipeline`: Interface for a pipeline. Allows binding the pipeline, descriptor sets and pushing constants.
- `ComputePipeline`: A concrete implementation of `Pipeline` for compute pipelines.

### Example

```c++
auto shader = std::make_shared<merian::ShaderModule>(context, "raytrace.comp.glsl.spv", loader);
auto pipeline_layout = merian::PipelineLayoutBuilder(context)
                           .add_descriptor_set_layout(desc_layout)
                           .build_pipeline_layout();
auto spec_builder = merian::SpecializationInfoBuilder();
spec_builder.add_entry(local_size_x, local_size_y); // constant ids 0 and 1
auto spec_info = spec_builder.build();
auto pipeline = merian::ComputePipeline(pipeline_layout, shader, spec_info);

cmd = pool->create_and_begin();
pipeline.bind(cmd);
pipeline.bind_descriptor_set(cmd, desc_set);

cmd.dispatch((uint32_t(width) + local_size_x - 1) / local_size_x,
             (uint32_t(height) + local_size_y - 1) / local_size_y, 1);
merian::cmd_barrier_compute_host(cmd);
pool->end_all();
queue->submit_wait(pool);
pool->reset();
```
