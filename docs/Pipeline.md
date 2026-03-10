## Pipelines

Merian provides builders for `PipelineLayout` and `SpecializationInfo`, as well as concrete pipeline types. Objects remember their creation parameters so you do not need to keep intermediate handles alive.

- `SpecializationInfoBuilder`: Builder for `SpecializationInfo`. Entries are added with auto-incrementing constant IDs via `add_entry()` or explicit IDs via `add_entry_id()`.
- `SpecializationInfo`: Stores specialization constant values and the `vk::SpecializationInfo`.
- `PipelineLayoutBuilder`: Fluent builder for `PipelineLayout`.
- `PipelineLayout`: Wraps a `vk::PipelineLayout` and destroys it in its destructor.
- `Pipeline`: Interface for all pipelines. Allows binding the pipeline, descriptor sets and pushing constants via the `CommandBuffer` API.
- `ComputePipeline`: Concrete `Pipeline` for compute passes. Created via `ComputePipeline::create()`.
- `GraphicsPipelineBuilder` / `GraphicsPipeline`: Builder and concrete `Pipeline` for graphics passes.

### Entry points

Shaders are represented as `EntryPointHandle` objects. Each entry point wraps a compiled `ShaderModule` and carries stage and name information. Entry points can be specialized:

```c++
auto specialized = entry_point->specialize(spec_info);
```

### Slang shaders

Slang shaders are compiled via a `SlangSession`. For simple cases there is a shortcut:

```c++
// Context already holds a ShaderCompileContext with the project's search paths.
auto compile_ctx = ShaderCompileContext::create(context);
auto session     = SlangSession::create(compile_ctx);

// Simple path: load module + compose + link + compile in one call.
EntryPointHandle entry_point = session->load_module_from_path_and_compile_entry_point(
    context, "my_shader.slang", "main");
```

For complex compositions (multiple modules, type conformances, specialization) use `SlangComposition` and `SlangSession::compose()`.

### Example (Slang compute pipeline)

```c++
auto compile_ctx = ShaderCompileContext::create(context);
auto session     = SlangSession::create(compile_ctx);

EntryPointHandle entry_point = session->load_module_from_path_and_compile_entry_point(
    context, "my_shader.slang");

// Specialization constants
// add_entry(T) returns uint32_t id; variadic add_entry(T...) returns std::vector<uint32_t>
auto spec_builder = merian::SpecializationInfoBuilder();
auto ids          = spec_builder.add_entry(local_size_x, local_size_y);
uint32_t id_x = ids[0], id_y = ids[1];
auto spec_info    = spec_builder.build();

// Pipeline layout
auto pipeline_layout = merian::PipelineLayoutBuilder(context)
    .add_descriptor_set_layout(desc_layout)
    .build_pipeline_layout();

// Pipeline
auto pipeline = merian::ComputePipeline::create(
    pipeline_layout,
    entry_point->specialize(spec_info)
);

// Recording
cmd->bind(pipeline);
cmd->bind_descriptor_set(pipeline, 0, desc_set);
cmd->dispatch(extent, local_size_x, local_size_y);
```

### Example (GLSL/SPIR-V compute pipeline)

```c++
auto shader = std::make_shared<merian::ShaderModule>(
    context, "shader.comp.spv", file_loader);

auto spec_builder = merian::SpecializationInfoBuilder();
spec_builder.add_entry(local_size_x, local_size_y); // constant IDs 0 and 1
auto spec_info = spec_builder.build();

auto pipeline_layout = merian::PipelineLayoutBuilder(context)
    .add_descriptor_set_layout(desc_layout)
    .build_pipeline_layout();

auto pipeline = merian::ComputePipeline::create(
    pipeline_layout,
    shader->get_entry_point()->specialize(spec_info)
);
```
