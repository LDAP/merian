#include "merian-graph/connectors/vk_tlas_in.hpp"
#include "merian-graph/graph/errors.hpp"

namespace merian {

VkTLASIn::VkTLASIn() = default;

void VkTLASIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<VkTLASOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{"VkTLASIn cannot receive from output."};
    }
}

const AccelerationStructureHandle& VkTLASIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<TLASResource>(resource)->tlas;
}

void VkTLASIn::bind(ShaderCursor& cursor,
                    const GraphResourceHandle& resource,
                    [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                    [[maybe_unused]] const ConnectorAccess& access) {
    cursor.write(debugable_ptr_cast<TLASResource>(resource)->tlas);
}

VkTLASInHandle VkTLASIn::create() {
    return std::make_shared<VkTLASIn>();
}

} // namespace merian
