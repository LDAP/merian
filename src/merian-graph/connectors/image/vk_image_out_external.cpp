#include "merian-graph/connectors/image/vk_image_out_external.hpp"

#include "merian/utils/pointer.hpp"

namespace merian {

ExternalVkImageOut::ExternalVkImageOut() = default;

GraphResourceHandle ExternalVkImageOut::create_resource(const ResourceCreateInfo& info) {
    return std::make_shared<ExternalImageArrayResource>(
        info.allocator->get_sampler_pool()->nearest_clamp_to_edge());
}

ExternalImageArrayResource& ExternalVkImageOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<ExternalImageArrayResource>(resource);
}

ExternalVkImageOutHandle ExternalVkImageOut::create() {
    return std::make_shared<ExternalVkImageOut>();
}

} // namespace merian
