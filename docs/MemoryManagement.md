## Memory management

### Interfaces

Merian provides several abstract interfaces for memory allocation:

- `SamplerPool`: Reuse samplers with identical configuration (keeps reference count).
- `MemoryAllocator`: Interface for memory allocators.
- `MemoryAllocation`: Interface for a memory allocation created by a `MemoryAllocator`.

### Implementation and utils

Merian provides concrete implementations of these interfaces:

- `MemoryAllocatorVMA`: An implementation of `MemoryAllocator` using the [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (VMA).
- `ResourceAllocator`: Uses a `MemoryAllocator` to create and destroy resources.

### Resource allocations

Current resources (that can be allocated using a `ResourceAllocator`):

- `Buffer`
- `Image`
- `Texture`: An image together with an image view and sampler.
- `AccelerationStructure`: For ray tracing.

Note that the destructor of these allocations destroys the underlying Vulkan resource and frees the associated memory.

### Memory mapping types

```cpp
enum class MemoryMappingType {
    NONE,                        // GPU-only memory
    HOST_ACCESS_RANDOM,          // CPU random access (readback)
    HOST_ACCESS_SEQUENTIAL_WRITE // CPU sequential write (upload)
};
```

### General example

```c++
int main() {
    merian::ContextHandle context = merian::Context::create({
        .context_extensions = {"merian-resources", "merian-debug-utils"},
    });
    auto alloc = context->get_context_extension<merian::ExtensionResources>()->resource_allocator();

    const uint32_t width = 800;
    const uint32_t height = 600;

    merian::BufferHandle result = alloc->create_buffer(
        width * height * 3 * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer,
        merian::MemoryMappingType::HOST_ACCESS_RANDOM
    );

    // Render...

    auto data = result->get_memory()->map();
    stbi_write_hdr("out.hdr", width, height, 3, reinterpret_cast<float*>(data));
    result->get_memory()->unmap();

    // Cleans up automatically (buffer → resource allocator → memory allocator → ... → context)
}
```

### Staging memory

The `ResourceAllocator` uses a `StagingMemoryManager` for uploads and downloads to and from device-local storage. Staging memory is automatically released when the command pool resets.

```c++
// Upload data in a command buffer
merian::BufferHandle gpu_buf = alloc->create_buffer(
    cmd, data.size() * sizeof(float),
    vk::BufferUsageFlagBits::eStorageBuffer,
    data.data()
);
```
