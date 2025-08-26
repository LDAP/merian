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
    switch (vulkan_api_version) {
    case VK_API_VERSION_1_0:
        return CompilationTarget::SPIRV_1_0;
    case VK_API_VERSION_1_1:
        return CompilationTarget::SPIRV_1_3;
    case VK_API_VERSION_1_2:
        return CompilationTarget::SPIRV_1_5;
    case VK_API_VERSION_1_3:
    case VK_API_VERSION_1_4:
        return CompilationTarget::SPIRV_1_6;

    default: {
        throw std::invalid_argument("unknown Vulkan API version");
    }
    }
}

class CompilationSessionDescription {
  public:
    CompilationSessionDescription() {}

    CompilationSessionDescription(
        const std::vector<std::filesystem::path>& include_paths = {},
        const std::map<std::string, std::string>& preprocessor_defines = {},
        const bool generate_debug_info = Context::IS_DEBUG_BUILD,
        const uint32_t optimization_level = Context::BUILD_OPTIMIZATION_LEVEL,
        const CompilationTarget target = CompilationTarget::SPIRV_1_6,
        const uint32_t target_vk_api_version = VK_API_VERSION_1_4)
        : include_paths(include_paths), preprocessor_defines(preprocessor_defines),
          debug_info(generate_debug_info), optimization_level(optimization_level), target(target),
          target_vk_api_version(target_vk_api_version) {}

    CompilationSessionDescription(const ContextHandle& context)
        : include_paths(context->get_default_shader_include_paths()),
          preprocessor_defines(context->get_default_shader_macro_definitions()),
          debug_info(Context::IS_DEBUG_BUILD),
          optimization_level(Context::BUILD_OPTIMIZATION_LEVEL),
          target(spirv_target_for_vulkan_api_version(context->vk_api_version)),
          target_vk_api_version(context->vk_api_version) {}

    // -------------------------------------------------

    void add_include_path(const std::filesystem::path& path) {
        include_paths.emplace_back(std::filesystem::weakly_canonical(path));
    }

    void add_include_path(const std::string& path) {
        include_paths.emplace_back(std::filesystem::weakly_canonical(path));
    }

    bool remove_include_path(const std::filesystem::path& path) {
        return remove_canonical_include_path(std::filesystem::weakly_canonical(path));
    }

    bool remove_include_path(const std::string& path) {
        return remove_canonical_include_path(std::filesystem::weakly_canonical(path));
    }

    void set_processor_define(const std::string& key, const std::string& value) {
        preprocessor_defines.emplace(key, value);
    }

    // Returns true if a define was unset
    bool unset_processor_define(const std::string& key) {
        return preprocessor_defines.erase(key) > 0;
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

    const std::vector<std::filesystem::path>& get_include_paths() const {
        return include_paths;
    }

    const std::map<std::string, std::string>& get_preprocessor_defines() const {
        return preprocessor_defines;
    }

    const bool& should_generate_debug_info() const {
        return debug_info;
    }

    const uint32_t& get_optimization_level() const {
        return optimization_level;
    }

    const CompilationTarget& get_target() {
        return target;
    }

    const uint32_t& get_target_vk_api_version() const {
        return target_vk_api_version;
    }

  private:
    bool remove_canonical_include_path(const std::filesystem::path& canonical) {
        bool removed = false;
        for (uint32_t i = 0; i < include_paths.size();) {
            if (include_paths[i] == canonical) {
                std::swap(include_paths.back(), include_paths[i]);
                include_paths.pop_back();
                removed = true;
            } else {
                i++;
            }
        }

        return removed;
    }

  private:
    std::vector<std::filesystem::path> include_paths;
    std::map<std::string, std::string> preprocessor_defines;
    bool debug_info;
    uint32_t optimization_level;
    CompilationTarget target;
    uint32_t target_vk_api_version;
};

} // namespace merian
