#include <gtest/gtest.h>

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/memory/memory_allocator_tracking.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

using namespace merian;

namespace {

constexpr vk::DeviceSize MIB = vk::DeviceSize{1024} * 1024;

} // namespace

TEST(TrackingMemoryAllocator, CountsLiveAllocations) {
    const ContextHandle context = Context::create({
        .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
        .application_name = "test-memory-allocator-tracking",
    });
    const ResourceAllocatorHandle base =
        context->get_context_extension<ExtensionResources>()->resource_allocator();
    const TrackingMemoryAllocatorHandle tracking =
        TrackingMemoryAllocator::create(base->get_memory_allocator());
    const ResourceAllocatorHandle allocator = std::make_shared<ResourceAllocator>(base, tracking);

    BufferHandle a = allocator->create_buffer(MIB, vk::BufferUsageFlagBits::eStorageBuffer);
    const BufferHandle b =
        allocator->create_buffer(2 * MIB, vk::BufferUsageFlagBits::eStorageBuffer);
    const ImageHandle c =
        allocator->create_image(vk::ImageCreateInfo{{},
                                                    vk::ImageType::e2D,
                                                    vk::Format::eR8G8B8A8Unorm,
                                                    {256, 256, 1},
                                                    1,
                                                    1,
                                                    vk::SampleCountFlagBits::e1,
                                                    vk::ImageTiling::eOptimal,
                                                    vk::ImageUsageFlagBits::eSampled});
    EXPECT_EQ(tracking->get_allocated_size(), a->get_memory_requirements().size +
                                                  b->get_memory_requirements().size +
                                                  c->get_memory_requirements().size);

    a.reset();
    EXPECT_EQ(tracking->get_allocated_size(),
              b->get_memory_requirements().size + c->get_memory_requirements().size);

    const BufferHandle untracked =
        base->create_buffer(MIB, vk::BufferUsageFlagBits::eStorageBuffer);
    EXPECT_EQ(tracking->get_allocated_size(),
              b->get_memory_requirements().size + c->get_memory_requirements().size);
}
