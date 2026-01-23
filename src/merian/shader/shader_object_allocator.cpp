#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/shader_object.hpp"

namespace merian {

DescriptorSetShaderObjectAllocator::DescriptorSetShaderObjectAllocator(
    const ResourceAllocatorHandle& allocator, const uint32_t iterations_in_flight)
    : allocator(allocator), iterations_in_flight(iterations_in_flight) {}

DescriptorContainerHandle
DescriptorSetShaderObjectAllocator::get_or_create_descriptor_set(const ShaderObjectHandle& object) {
    auto it = sets.find(object);

    if (it == sets.end()) {
        const DescriptorSetLayoutHandle layout = create_descriptor_set_layout_from_slang_type_layout(
            allocator->get_context(), object->get_type_layout());

        std::tie(it, std::ignore) =
            sets.emplace(object, allocator->allocate_descriptor_set(layout, iterations_in_flight));
    }

    return it->second[iteration_in_flight];
}

void DescriptorSetShaderObjectAllocator::set_iteration(const uint32_t iteration) {
    iteration_in_flight = iteration;
}

void DescriptorSetShaderObjectAllocator::reset() {
    sets.clear();
}

} // namespace merian
