#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/shader_object.hpp"

namespace merian {

// --- SimpleShaderObjectAllocator ---

SimpleShaderObjectAllocator::SimpleShaderObjectAllocator(const ResourceAllocatorHandle& allocator)
    : allocator(allocator) {}

DescriptorContainerHandle SimpleShaderObjectAllocator::allocate(ShaderObject* object) {
    return allocator->allocate_descriptor_set(
        object->get_object_layout()->get_descriptor_set_layout());
}

void SimpleShaderObjectAllocator::free(ShaderObject*) {}

BufferHandle SimpleShaderObjectAllocator::allocate_uniform_buffer(const vk::DeviceSize size) {
    return allocator->create_buffer(size, vk::BufferUsageFlagBits::eUniformBuffer |
                                              vk::BufferUsageFlagBits::eTransferDst);
}

StagingMemoryManagerHandle SimpleShaderObjectAllocator::get_staging() const {
    return allocator->get_staging();
}

// --- FrameCachingShaderObjectAllocator ---

FrameCachingShaderObjectAllocator::FrameCachingShaderObjectAllocator(
    const ResourceAllocatorHandle& allocator, const uint32_t iterations_in_flight)
    : allocator(allocator), iterations_in_flight(iterations_in_flight) {}

DescriptorContainerHandle FrameCachingShaderObjectAllocator::allocate(ShaderObject* object) {
    auto it = cache.find(object);

    if (it == cache.end()) {
        std::tie(it, std::ignore) =
            cache.emplace(object, allocator->allocate_descriptor_set(
                                      object->get_object_layout()->get_descriptor_set_layout(),
                                      iterations_in_flight));
    }

    return it->second[current_iteration];
}

void FrameCachingShaderObjectAllocator::free(ShaderObject* object) {
    cache.erase(object);
}

BufferHandle FrameCachingShaderObjectAllocator::allocate_uniform_buffer(const vk::DeviceSize size) {
    return allocator->create_buffer(size, vk::BufferUsageFlagBits::eUniformBuffer |
                                              vk::BufferUsageFlagBits::eTransferDst);
}

StagingMemoryManagerHandle FrameCachingShaderObjectAllocator::get_staging() const {
    return allocator->get_staging();
}

void FrameCachingShaderObjectAllocator::set_iteration(const uint32_t iteration) {
    current_iteration = iteration % iterations_in_flight;
}

void FrameCachingShaderObjectAllocator::reset() {
    cache.clear();
}

} // namespace merian
