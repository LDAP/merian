## Memory management

### Interfaces

Merian provides several abstract interfaces for memory allocation:

- `SamplerPool`: Reuse samplers with identical configuration (keeps reference count).
- `MemoryAllocator`: Interface for memory allocators. 
- `MemoryAllocation`: Interface for a memory allocation created by a `MemoryAllocator`.

### Implementation and Utils

Merian provides concrete implementations of these interfaces:

- `MemoryAllocatorVMA`: A implementation of `MemoryAllocator` using the [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (VMA).
- `ResourceAllocator`: Uses a `MemoryAllocator` to create and destroy resources.

### Resource allocations

Current resources (that can be allocated using a ResourceAllocator):

- `Buffer`
- `Image`
- `Texture`: A image together with an image view and sampler
- `AccelerationStructure`: for ray tracing

Note that the destructor of these allocations destroys the underlying Vulkan resource and frees the associated memory.

### General example

```c++
int main() {
    const auto debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(false);
    const auto resources = std::make_shared<merian::ExtensionResources>();
    const std::vector<std::shared_ptr<merian::Extension>> extensions = {resources, debug_utils};

    const merian::ContextHandle context = merian::Context::make_context(extensions, "merian");
    auto alloc = resources->resource_allocator();

    const uint32_t width = 800;
    const uint32_t height = 600;

    merian::BufferHandle result = alloc->createBuffer(
        width * height * 3 * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer,
        merian::HOST_ACCESS_RANDOM
    );

    // Render...

    auto data = result->get_memory()->map();
    stbi_write_hdr("out.hdr", width, height, 3, reinterpret_cast<float*>(data));
    result->get_memory()->unmap();

    // Cleans up automagically (first buffer then resource allocator then memory allocator then, ..., then context)
}
```

### Staging memory

The `ResourceAllocator` uses a `StagingMemoryManager` for up and downloads to and from device local storage. The staging memory manager automatically releases stating memory when the command buffer pool resets and releases all resources.
