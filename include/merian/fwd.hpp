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

} // namespace merian
