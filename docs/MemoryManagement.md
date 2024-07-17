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

The `ResourceAllocator` uses a `StagingMemoryManager` for up and downloads to and from device local storage.
This staging area must be released periodically.
Therefore, you must call `finalizeResources()` start a new resource set and `releaseResources()` to release
resources for finished transfers.

Since transfers are asynchronous you must ensure that the transfers have been finished.
The `StagingMemoryManager` provides two methods for this: A fence that can be supplied when finalizing
or a resource set ID can be retrieved to free the resources manually.

Example:

```c++
struct FrameData {
    merian::StagingMemoryManager::SetID staging_set_id{};
};

// ensures that we can release the resources
auto ring_fences = make_shared<merian::RingFences<3, FrameData>>(context);

while (!glfwWindowShouldClose(*window)) {
    auto frame_data = ring_fences->wait_and_get();
    // Releases the resources for this ring-iteration.
    // We are sure that these have been transfered, since we submitted using the fence.
    alloc->getStaging()->releaseResourceSet(frame_data.user_data.staging_set_id);

    // allocate resources, issue uploads, downloads...

    // Finalize the resource set for this frame and start a new set for the next frame
    frame_data.user_data.staging_set_id = alloc->getStaging()->finalizeResourceSet();
    // Submit using the fence from our RingFences.
    queue->submit(..., frame_data.fence);
}
```
