#pragma once

#include <memory>

namespace merian {

// Core classes
class Context;
class Instance;
class PhysicalDevice;
class Device;

// Handle types for Context-managed objects
using ContextHandle = std::shared_ptr<Context>;
using WeakContextHandle = std::weak_ptr<Context>;
using InstanceHandle = std::shared_ptr<Instance>;
using WeakInstanceHandle = std::weak_ptr<Instance>;
using PhysicalDeviceHandle = std::shared_ptr<PhysicalDevice>;
using WeakPhysicalDeviceHandle = std::weak_ptr<PhysicalDevice>;
using DeviceHandle = std::shared_ptr<Device>;
using WeakDeviceHandle = std::weak_ptr<Device>;

// Command related
class Queue;
using QueueHandle = std::shared_ptr<Queue>;
class CommandPool;
using CommandPoolHandle = std::shared_ptr<CommandPool>;
class CommandBuffer;
using CommandBufferHandle = std::shared_ptr<CommandBuffer>;

// Shader
class SlangSession;
using SlangSessionHandle = std::shared_ptr<SlangSession>;

// New extension hierarchy (in extension/*.hpp)
class ContextExtension;

// Memory & Resources
class Resource;
class Sampler;
class Buffer;
class Image;
class ImageView;
class Texture;
class AccelerationStructure;
using ResourceHandle = std::shared_ptr<Resource>;
using SamplerHandle = std::shared_ptr<Sampler>;
using BufferHandle = std::shared_ptr<Buffer>;
using ImageHandle = std::shared_ptr<Image>;
using ImageViewHandle = std::shared_ptr<ImageView>;
using TextureHandle = std::shared_ptr<Texture>;
using AccelerationStructureHandle = std::shared_ptr<AccelerationStructure>;

// Descriptors
class DescriptorSet;
class DescriptorSetLayout;
class DescriptorPool;
class DescriptorBuffer;
using DescriptorSetHandle = std::shared_ptr<DescriptorSet>;
using DescriptorSetLayoutHandle = std::shared_ptr<DescriptorSetLayout>;
using DescriptorPoolHandle = std::shared_ptr<DescriptorPool>;
using DescriptorBufferHandle = std::shared_ptr<DescriptorBuffer>;

// Pipelines
class Pipeline;
class PipelineGraphics;
class PipelineCompute;
class PipelineLayout;
using PipelineHandle = std::shared_ptr<Pipeline>;
using PipelineGraphicsHandle = std::shared_ptr<PipelineGraphics>;
using PipelineComputeHandle = std::shared_ptr<PipelineCompute>;
using PipelineLayoutHandle = std::shared_ptr<PipelineLayout>;

// Renderpass
class Renderpass;
class Framebuffer;
using RenderpassHandle = std::shared_ptr<Renderpass>;
using FramebufferHandle = std::shared_ptr<Framebuffer>;

// Synchronization
class Semaphore;
class SemaphoreBinary;
class SemaphoreTimeline;
class Fence;
using SemaphoreHandle = std::shared_ptr<Semaphore>;
using SemaphoreBinaryHandle = std::shared_ptr<SemaphoreBinary>;
using SemaphoreTimelineHandle = std::shared_ptr<SemaphoreTimeline>;
using FenceHandle = std::shared_ptr<Fence>;

// Shader System
class ShaderModule;
class ShaderObject;
class EntryPoint;
class SlangProgram;
class SlangEntryPoint;
class SlangComposition;
class ShaderCompileContext;
using ShaderModuleHandle = std::shared_ptr<ShaderModule>;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;
using EntryPointHandle = std::shared_ptr<EntryPoint>;
using SlangProgramHandle = std::shared_ptr<SlangProgram>;
using SlangEntryPointHandle = std::shared_ptr<SlangEntryPoint>;
using SlangCompositionHandle = std::shared_ptr<SlangComposition>;
using ShaderCompileContextHandle = std::shared_ptr<ShaderCompileContext>;

// Memory Allocation
class MemoryAllocation;
using MemoryAllocationHandle = std::shared_ptr<MemoryAllocation>;

} // namespace merian
