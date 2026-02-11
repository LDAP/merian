#pragma once

#include "merian/vk/context.hpp"
#include "vulkan/vulkan_core.h"

#include <cstdint>
#include <stdexcept>

namespace merian {

enum class CompilationTarget {
    SPIRV_1_0,
    SPIRV_1_1,
    SPIRV_1_2,
    SPIRV_1_3,
    SPIRV_1_4,
    SPIRV_1_5,
    SPIRV_1_6,
};

inline CompilationTarget spirv_target_for_vulkan_api_version(const uint32_t vulkan_api_version) {
    if (vulkan_api_version >= VK_API_VERSION_1_3) {
        return CompilationTarget::SPIRV_1_6;
    }
    if (vulkan_api_version >= VK_API_VERSION_1_2) {
        return CompilationTarget::SPIRV_1_5;
    }
    if (vulkan_api_version >= VK_API_VERSION_1_1) {
        return CompilationTarget::SPIRV_1_3;
    }
    return CompilationTarget::SPIRV_1_0;
}

class ShaderCompileContext;
using ShaderCompileContextHandle = std::shared_ptr<ShaderCompileContext>;

class ShaderCompileContext {
  protected:
    ShaderCompileContext(const vk::ArrayProxy<std::filesystem::path>& search_paths = {},
                         const std::map<std::string, std::string>& preprocessor_macros = {},
                         const bool generate_debug_info = Context::IS_DEBUG_BUILD,
                         const uint32_t optimization_level = Context::BUILD_OPTIMIZATION_LEVEL,
                         const CompilationTarget target = CompilationTarget::SPIRV_1_6,
                         const uint32_t target_vk_api_version = VK_API_VERSION_1_4)
        : preprocessor_macros(preprocessor_macros), debug_info(generate_debug_info),
          optimization_level(optimization_level), target(target),
          target_vk_api_version(target_vk_api_version) {
        file_loader.add_search_path(search_paths);
    }

    ShaderCompileContext(const ContextHandle& context)
        : preprocessor_macros(context->get_device()->get_shader_defines()),
          debug_info(Context::IS_DEBUG_BUILD),
          optimization_level(Context::BUILD_OPTIMIZATION_LEVEL),
          target(spirv_target_for_vulkan_api_version(context->get_device()->get_vk_api_version())),
          target_vk_api_version(context->get_device()->get_vk_api_version()) {
        // Add search paths from context's file loader
        const auto& context_file_loader = context->get_file_loader();
        for (const auto& path : *context_file_loader) {
            file_loader.add_search_path(path);
        }
    }

  public:
    // -------------------------------------------------

    void add_search_path(const std::filesystem::path& path) {
        file_loader.add_search_path(path);
    }

    bool remove_include_path(const std::filesystem::path& path) {
        return file_loader.remove_search_path(path);
    }

    void set_preprocessor_macro(const std::string& key, const std::string& value) {
        preprocessor_macros.emplace(key, value);
    }

    std::string& operator[](const std::string& key) {
        return preprocessor_macros[key];
    }

    std::string& operator[](const std::string&& key) {
        return preprocessor_macros[key];
    }

    void set_preprocessor_macros(const std::map<std::string, std::string>& key_value_map) {
        preprocessor_macros.insert(key_value_map.begin(), key_value_map.end());
    }

    // Returns true if a define was unset
    bool unset_preprocessor_macro(const std::string& key) {
        return preprocessor_macros.erase(key) > 0;
    }

    void set_geneate_debug_info(const bool enable) {
        debug_info = enable;
    }

    void set_optimization_level(const uint32_t level) {
        assert(level <= 3);
        optimization_level = level;
    }

    void set_target(const CompilationTarget target) {
        this->target = target;
    }

    void set_target_vk_api_version(const uint32_t target_vk_api_version) {
        this->target_vk_api_version = target_vk_api_version;
    }

    // -------------------------------------------------

    const std::map<std::string, std::string>& get_preprocessor_macros() const {
        return preprocessor_macros;
    }

    const bool& should_generate_debug_info() const {
        return debug_info;
    }

    const uint32_t& get_optimization_level() const {
        return optimization_level;
    }

    const CompilationTarget& get_target() const {
        return target;
    }

    const uint32_t& get_target_vk_api_version() const {
        return target_vk_api_version;
    }

    const FileLoader& get_search_path_file_loader() const {
        return file_loader;
    }

    // -------------------------------------------------

    static ShaderCompileContextHandle create(const ContextHandle& context) {
        return ShaderCompileContextHandle(new ShaderCompileContext(context));
    }

    static ShaderCompileContextHandle
    create(const vk::ArrayProxy<std::filesystem::path>& search_paths,
           const PhysicalDeviceHandle& physical_device) {
        auto context = ShaderCompileContextHandle(new ShaderCompileContext(
            search_paths, physical_device->get_shader_defines(), Context::IS_DEBUG_BUILD,
            Context::BUILD_OPTIMIZATION_LEVEL,
            spirv_target_for_vulkan_api_version(physical_device->get_vk_api_version()),
            physical_device->get_vk_api_version()));
        return context;
    }

    static ShaderCompileContextHandle
    create(const vk::ArrayProxy<std::filesystem::path>& search_paths, const DeviceHandle& device) {
        auto context = ShaderCompileContextHandle(new ShaderCompileContext(
            search_paths, device->get_shader_defines(), Context::IS_DEBUG_BUILD,
            Context::BUILD_OPTIMIZATION_LEVEL,
            spirv_target_for_vulkan_api_version(device->get_vk_api_version()),
            device->get_vk_api_version()));
        return context;
    }

  private:
    FileLoader file_loader; // for search path management

    std::map<std::string, std::string> preprocessor_macros;
    bool debug_info;
    uint32_t optimization_level;
    CompilationTarget target;
    uint32_t target_vk_api_version;
};

} // namespace merian
