#pragma once

#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <vector>

namespace merian {

// Creates descriptor sets with from the cmd_build inputs.
//
// An appropriate layout is created if optional_layout is null.
// The graph resources are bound in order input images, input buffers, output images,
// output buffers.
// Input images are bound as sampler2d, output images as image2d.
// You need to keep all returned resources alive, else the descriptors get invalid.
[[nodiscard]] std::tuple<std::vector<TextureHandle>,
                         std::vector<DescriptorSetHandle>,
                         DescriptorPoolHandle,
                         DescriptorSetLayoutHandle>
make_graph_descriptor_sets(const SharedContext context,
                           const ResourceAllocatorHandle allocator,
                           const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                           const std::vector<std::vector<merian::BufferHandle>>& buffer_inputs,
                           const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                           const std::vector<std::vector<merian::BufferHandle>>& buffer_outputs,
                           const DescriptorSetLayoutHandle optional_layout);

} // namespace merian
