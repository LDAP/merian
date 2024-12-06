#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/utils/concurrent/thread_pool.hpp"
#include <map>
#include <spdlog/logger.h>

#include <typeindex>

// attempt to set the dynamic dispach launcher as early as possible
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif

#include <vulkan/vulkan.hpp>

namespace merian {

// cyclic -> forward definition
class Extension;
class Context;
class Queue;
class CommandPool;

using ContextHandle = std::shared_ptr<Context>;
class ShaderModule;
using ShaderModuleHandle = std::shared_ptr<ShaderModule>;
class ShaderCompiler;
using ShaderCompilerHandle = std::shared_ptr<ShaderCompiler>;

struct PhysicalDevice {
    const vk::PhysicalDevice& operator*() const {
        return physical_device;
    }

    operator const vk::PhysicalDevice&() const {
        return physical_device;
    }

    operator vk::PhysicalDevice&() {
        return physical_device;
    }

    const vk::PhysicalDeviceLimits& get_physical_device_limits() const {
        return physical_device_properties.properties.limits;
    }

    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties2 physical_device_properties;
    // all supported features. Must be enabled with ExtensionVkCore.
    vk::PhysicalDeviceFeatures2 physical_device_features;
    vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;
    vk::PhysicalDeviceSubgroupProperties physical_device_subgroup_properties;
    std::vector<vk::ExtensionProperties> physical_device_extension_properties;
};

class ExtensionContainer {
  public:
    template <class Extension> std::shared_ptr<Extension> get_extension() const {
        if (extensions.contains(typeid(Extension))) {
            return std::static_pointer_cast<Extension>(extensions.at(typeid(Extension)));
        }
        return nullptr;
    }

  protected:
    std::unordered_map<std::type_index, std::shared_ptr<Extension>> extensions;
};

struct QueueInfo {
    // A queue family index guaranteed to support graphics+compute+transfer (or -1)
    int32_t queue_family_idx_GCT = -1;
    // A queue family index guaranteed to support compute (or -1)
    int32_t queue_family_idx_C = -1;
    // A queue family index guaranteed to support transfer (or -1)
    int32_t queue_family_idx_T = -1;

    // The queue indices (not family!) used for the graphics queue
    int32_t queue_idx_GCT = -1;
    // The queue indices (not family!) used for the compute queue
    std::vector<uint32_t> queue_idx_C;
    // The queue indices (not family!) used for the transfer queue
    int32_t queue_idx_T = -1;
};

/* Initializes the Vulkan instance and device and holds core objects.
 *
 * Extensions can extend the functionality and hook into the creation process.
 */
class Context : public std::enable_shared_from_this<Context>, public ExtensionContainer {

  public:
#ifdef NDEBUG
    static constexpr bool IS_DEBUG_BUILD = false;
#else
    static constexpr bool IS_DEBUG_BUILD = true;
#endif

    /**
     * @brief      Use this method to create the context.
     *
     */
    static ContextHandle
    create(const std::vector<std::shared_ptr<Extension>>& extensions,
           const std::string& application_name = "",
           const uint32_t application_vk_version = VK_MAKE_VERSION(1, 0, 0),
           const uint32_t preffered_number_compute_queues = 1, // Additionally to the GCT queue
           const uint32_t vk_api_version = VK_API_VERSION_1_3,
           const bool require_extension_support = false,
           const uint32_t filter_vendor_id = -1,
           const uint32_t filter_device_id = -1,
           const std::string& filter_device_name = "");

  private:
    Context(const std::vector<std::shared_ptr<Extension>>& extensions,
            const std::string& application_name,
            const uint32_t application_vk_version,
            const uint32_t preffered_number_compute_queues,
            const uint32_t vk_api_version,
            const bool require_extension_support,
            const uint32_t filter_vendor_id,
            const uint32_t filter_device_id,
            const std::string& filter_device_name);

  public:
    ~Context();

    operator vk::Instance&() {
        return instance;
    }

    operator vk::PhysicalDevice&() {
        return physical_device;
    }

    operator vk::Device&() {
        return device;
    }

  private: // Vulkan initialization
    void create_instance();
    void prepare_physical_device(uint32_t filter_vendor_id,
                                 uint32_t filter_device_id,
                                 std::string filter_device_name);
    void find_queues();
    void create_device_and_queues(uint32_t preffered_number_compute_queues);
    void prepare_shader_include_defines();

  private: // Helper
    void extensions_check_instance_layer_support(const bool fail_if_unsupported);
    void extensions_check_instance_extension_support(const bool fail_if_unsupported);
    void extensions_check_device_extension_support(const bool fail_if_unsupported);
    void extensions_self_check_support(const bool fail_if_unsupported);
    void destroy_unsupported_extensions(const std::vector<std::shared_ptr<Extension>>& extensions,
                                        const bool fail_if_unsupported);

  public: // Getter
    // The actual number of compute queues (< preffered_number_compute_queues).
    uint32_t get_number_compute_queues() const noexcept;

    // A queue guaranteed to support graphics+compute+transfer.
    // Can be nullptr if unavailable, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<Queue> get_queue_GCT();

    // A queue guaranteed to support transfer.
    // Can be nullptr if unavailable, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed.
    // Might fall back to the GCT queue if fallback is true.
    std::shared_ptr<Queue> get_queue_T(const bool fallback = false);

    // A queue guaranteed to support compute.
    // Can be nullptr if unavailable, you can check that with if (shrd_ptr) {...}
    // Make sure to keep a reference, else the pool and its buffers are destroyed.
    // Might fall back to a different compute queue or the GCT queue, if fallback is true.
    // In this case index may be invalid, else it must be < get_number_compute_queues() and or
    // nullptr is returned.
    std::shared_ptr<Queue> get_queue_C(uint32_t index = 0, const bool fallback = false);

    // Convenience command pool for graphics and compute (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_GCT();

    // Convenience command pool for transfer (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_T();

    // Convenience command pool for compute (can be nullptr in very rare occasions)
    // Make sure to keep a reference, else the pool and its buffers are destroyed
    std::shared_ptr<CommandPool> get_cmd_pool_C();

    bool device_extension_enabled(const std::string& name) const;

    bool instance_extension_enabled(const std::string& name) const;

    const std::vector<const char*>& get_enabled_device_extensions() const;

    const std::vector<const char*>& get_enabled_instance_extensions() const;

    const std::vector<std::string>& get_default_shader_include_paths() const;

    const std::map<std::string, std::string>& get_default_shader_macro_definitions() const;

  private:
    // in create_instance

    std::vector<const char*> instance_layer_names;
    std::vector<const char*> instance_extension_names;

    // in create_device_and_queues

    std::vector<const char*> device_extensions;

  public:
    const std::string application_name;
    const uint32_t vk_api_version;
    const uint32_t application_vk_version;

    // in create_instance

    vk::Instance instance;

    // in prepare_physical_device

    // the vk::PhysicalDevice for this Context
    PhysicalDevice physical_device;

    // in create_device_and_queues

    // the vk::Device for this Context
    vk::Device device;

    vk::PipelineCache pipeline_cache;

    // -----------------
    // A shared thread pool with default size.
    ThreadPool thread_pool;

    // A shared file_loader for convenience.
    FileLoader file_loader;

    // A shader compiler with default include paths for convenience.
    ShaderCompilerHandle shader_compiler;

  private:
    // in find_queues. Indexes are -1 if no suitable queue was found!

    QueueInfo queue_info;

    // can be nullptr in very rare occasions, you can check that with if (shrd_ptr) {...}
    std::weak_ptr<Queue> queue_GCT;
    // can be nullptr in very rare occasions, you can check that with if (shrd_ptr) {...}
    std::weak_ptr<Queue> queue_T;
    // resized in create_device_and_queues
    std::vector<std::weak_ptr<Queue>> queues_C;

    // in make_context

    // Convenience command pool for graphics and compute (can be nullptr in very rare occasions)
    std::weak_ptr<CommandPool> cmd_pool_GCT;
    // Convenience command pool for transfer (can be nullptr in very rare occasions)
    std::weak_ptr<CommandPool> cmd_pool_T;
    // Convenience command pool for compute (can be nullptr in very rare occasions)
    std::weak_ptr<CommandPool> cmd_pool_C;

    std::vector<std::string> default_shader_include_paths;
    std::map<std::string, std::string> default_shader_macro_definitions;
};

} // namespace merian

#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
