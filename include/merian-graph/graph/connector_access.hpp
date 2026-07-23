#pragma once

#include <vulkan/vulkan.hpp>

namespace merian {

// Declarative: the graph emits the barriers from this, never the connector or node.
struct ConnectorAccess {
    vk::PipelineStageFlags2 stages{};
    vk::AccessFlags2 access{};
    vk::ImageUsageFlags image_usage{};
    vk::BufferUsageFlags buffer_usage{};

    bool is_write() const {
        constexpr vk::AccessFlags2 write_mask =
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite |
            vk::AccessFlagBits2::eColorAttachmentWrite |
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
            vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostWrite |
            vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eAccelerationStructureWriteKHR;
        return static_cast<bool>(access & write_mask);
    }

    bool empty() const {
        return !stages && !access;
    }

    friend ConnectorAccess operator|(const ConnectorAccess& a, const ConnectorAccess& b) {
        return {a.stages | b.stages, a.access | b.access, a.image_usage | b.image_usage,
                a.buffer_usage | b.buffer_usage};
    }

    static const ConnectorAccess compute_read;
    static const ConnectorAccess compute_write;
    static const ConnectorAccess compute_read_write;
    static const ConnectorAccess fragment_read;
    static const ConnectorAccess color_attachment;
    static const ConnectorAccess transfer_src;
    static const ConnectorAccess transfer_dst;
    static const ConnectorAccess ray_tracing_read;
    static const ConnectorAccess ray_tracing_write;
    static const ConnectorAccess ray_tracing_read_write;
    static const ConnectorAccess acceleration_structure_read;
};

inline const ConnectorAccess ConnectorAccess::compute_read{
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderRead,
    vk::ImageUsageFlagBits::eSampled,
    {}};

inline const ConnectorAccess ConnectorAccess::compute_write{
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderWrite,
    vk::ImageUsageFlagBits::eStorage,
    {}};

inline const ConnectorAccess ConnectorAccess::compute_read_write{
    vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
    {}};

inline const ConnectorAccess ConnectorAccess::fragment_read{
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderRead,
    vk::ImageUsageFlagBits::eSampled,
    {}};

inline const ConnectorAccess ConnectorAccess::color_attachment{
    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    vk::AccessFlagBits2::eColorAttachmentWrite,
    vk::ImageUsageFlagBits::eColorAttachment,
    {}};

inline const ConnectorAccess ConnectorAccess::transfer_src{
    vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferRead,
    vk::ImageUsageFlagBits::eTransferSrc, vk::BufferUsageFlagBits::eTransferSrc};

inline const ConnectorAccess ConnectorAccess::transfer_dst{
    vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite,
    vk::ImageUsageFlagBits::eTransferDst, vk::BufferUsageFlagBits::eTransferDst};

inline const ConnectorAccess ConnectorAccess::ray_tracing_read{
    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
    vk::AccessFlagBits2::eShaderRead,
    vk::ImageUsageFlagBits::eSampled,
    {}};

inline const ConnectorAccess ConnectorAccess::ray_tracing_write{
    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
    vk::AccessFlagBits2::eShaderWrite,
    vk::ImageUsageFlagBits::eStorage,
    {}};

inline const ConnectorAccess ConnectorAccess::ray_tracing_read_write{
    vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
    {}};

inline const ConnectorAccess ConnectorAccess::acceleration_structure_read{
    vk::PipelineStageFlagBits2::eRayTracingShaderKHR | vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    {},
    {}};

} // namespace merian
