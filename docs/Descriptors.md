## Descriptors

Merian provides a builder for `DescriptorSetLayouts` as well as a `DescriptorSetUpdate` utility class which eases binding descriptors to resources.

- `DescriptorSetLayoutBuilder`: A builder for `DescriptorSetLayout`.
- `DescriptorSetLayout`: Wraps a descriptor set layout and destroys it in its destructor. Makes it easy to generate a pool from this layout.
- `DescriptorPool`: Wraps a descriptor pool and destroys it in its destructor. Makes it easy to generate sets from this pool.
- `DescriptorSet`: Wraps a descriptor set and destroys it in its destructor.

### Example

```c++
auto builder = merian::DescriptorSetLayoutBuilder()
    .add_binding_storage_buffer()           // result
    .add_binding_acceleration_structure()   // top-level acceleration structure
    .add_binding_storage_buffer()           // vertex buffer
    .add_binding_storage_buffer();          // index buffer

auto desc_layout = builder.build_layout(context);
// Shared pointers are used -> allows automatic destruction in the right order.
auto desc_pool = std::make_shared<merian::DescriptorPool>(desc_layout);
auto desc_set = std::make_shared<merian::DescriptorSet>(desc_pool);

merian::DescriptorSetUpdate(desc_set)
    .write_descriptor_buffer(0, result)
    .write_descriptor_acceleration_structure(1, {as})
    .write_descriptor_buffer(2, vertex_buffer)
    .write_descriptor_buffer(3, index_buffer)
    .update(context);
```
