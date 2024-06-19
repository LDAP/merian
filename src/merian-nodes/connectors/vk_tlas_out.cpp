#include "vk_tlas_out.hpp"

#include "graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/utils/pointer.hpp"

namespace merian_nodes {

VkTLASOut::VkTLASOut(const std::string& name, const vk::ShaderStageFlags stage_flags)
    : TypedOutputConnector(name, false), stage_flags(stage_flags) {}

std::optional<vk::DescriptorSetLayoutBinding> VkTLASOut::get_descriptor_info() const {
    if (stage_flags)
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eAccelerationStructureKHR, 1,
                                              stage_flags, nullptr};
    return std::nullopt;
}

void VkTLASOut::get_descriptor_update(const uint32_t binding,
                                      GraphResourceHandle& resource,
                                      DescriptorSetUpdate& update) {
    const auto& res = debugable_ptr_cast<TLASResource>(resource);
    update.write_descriptor_acceleration_structure(binding, *res->tlas);
}

GraphResourceHandle VkTLASOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    [[maybe_unused]] const ResourceAllocatorHandle& allocator,
    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resoruce_index,
    const uint32_t ring_size) {

    return std::make_shared<TLASResource>(ring_size);
}

AccelerationStructureHandle& VkTLASOut::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<TLASResource>(resource)->tlas;
}

Connector::ConnectorStatusFlags
VkTLASOut::on_pre_process([[maybe_unused]] GraphRun& run,
                          [[maybe_unused]] const vk::CommandBuffer& cmd,
                          GraphResourceHandle& resource,
                          [[maybe_unused]] const NodeHandle& node,
                          [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                          [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TLASResource>(resource);

    if (!stage_flags || !res->tlas) {
        throw graph_errors::connector_error{
            fmt::format("Node {} must set the TLAS in connector {} in pre_process since "
                        "stage_flags are not empty",
                        node->name, name)};
    }

    if (res->needs_descriptor_update) {
        res->needs_descriptor_update = false;
        return NEEDS_DESCRIPTOR_UPDATE;
    }

    return {};
}

Connector::ConnectorStatusFlags VkTLASOut::on_post_process(
    GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TLASResource>(resource);

    if (!res->tlas) {
        throw graph_errors::connector_error{
            fmt::format("Node {} must set the TLAS in connector {}", node->name, name)};
    }

    Connector::ConnectorStatusFlags flags{};
    if (res->needs_descriptor_update) {
        res->needs_descriptor_update = false;
        flags |= NEEDS_DESCRIPTOR_UPDATE;
    }

    res->in_flight_tlas[run.get_in_flight_index()] = res->tlas;

    return flags;
}

VkTLASOutHandle VkTLASOut::compute_read(const std::string& name) {
    return std::make_shared<VkTLASOut>(name, vk::ShaderStageFlagBits::eCompute);
}

VkTLASOutHandle VkTLASOut::create(const std::string& name) {
    return std::make_shared<VkTLASOut>(name, vk::ShaderStageFlags{});
}

} // namespace merian_nodes
