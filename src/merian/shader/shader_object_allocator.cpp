#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/shader_object.hpp"

namespace merian {

SimpleShaderObjectAllocator::SimpleShaderObjectAllocator(const ResourceAllocatorHandle& allocator)
    : allocator(allocator) {}

DescriptorContainerHandle SimpleShaderObjectAllocator::allocate(const ShaderObjectHandle& object) {
    return allocator->allocate_descriptor_set(
        object->get_object_layout()->get_descriptor_set_layout());
}

FrameCachingShaderObjectAllocator::FrameCachingShaderObjectAllocator(
    const ResourceAllocatorHandle& allocator, const uint32_t iterations_in_flight)
    : allocator(allocator), iterations_in_flight(iterations_in_flight) {}

DescriptorContainerHandle
FrameCachingShaderObjectAllocator::allocate(const ShaderObjectHandle& object) {
    auto it = cache.find(object.get());

    if (it != cache.end()) {
        // Resolves pointer aliasing: prior owner died and a new ShaderObject was reborn here.
        if (auto live = it->second.live.lock(); live == object) {
            return it->second.sets[current_iteration];
        }
        cache.erase(it);
    }

    prune_expired();

    auto sets = allocator->allocate_descriptor_set(
        object->get_object_layout()->get_descriptor_set_layout(), iterations_in_flight);
    auto [inserted, _] = cache.try_emplace(object.get(), Entry{object, std::move(sets)});
    return inserted->second.sets[current_iteration];
}

void FrameCachingShaderObjectAllocator::set_iteration(const uint32_t iteration) {
    current_iteration = iteration % iterations_in_flight;
}

void FrameCachingShaderObjectAllocator::reset() {
    cache.clear();
}

void FrameCachingShaderObjectAllocator::prune_expired() {
    for (auto it = cache.begin(); it != cache.end();) {
        if (it->second.live.expired()) {
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace merian
