#pragma once

#include <memory>
#include <spdlog/spdlog.h>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

#define MERIAN_INITIAL_SPEC_CONSTANT_ALLOC_SIZE 32

/**
 * @brief      Wrapper for vk::SpecializationInfo, that holds the memory for specialization
 * constants
 *
 * Keep this object alive until the pipeline is created!
 */
class SpecializationInfo {

  public:
    SpecializationInfo() : entries() {}

    // You can free `data` directly
    SpecializationInfo(const std::vector<vk::SpecializationMapEntry>& entries,
                       const std::size_t data_size,
                       void* const data)
        : entries(entries) {

        this->data = new char[data_size];
        memcpy(this->data, data, data_size);

        if (data)
            info = vk::SpecializationInfo{static_cast<uint32_t>(this->entries.size()),
                                          this->entries.data(), data_size, this->data};
        else
            info = vk::SpecializationInfo{};
    }

  public:
    ~SpecializationInfo() {
        delete[] (data);
    }

  public:
    operator const vk::SpecializationInfo&() const {
        return info;
    }

    operator const vk::SpecializationInfo*() const {
        return &info;
    }

    const vk::SpecializationInfo& get() const {
        return info;
    }

  private:
    const std::vector<vk::SpecializationMapEntry> entries;
    char* data = nullptr;
    vk::SpecializationInfo info;
};

using SpecializationInfoHandle = std::shared_ptr<SpecializationInfo>;
static SpecializationInfoHandle MERIAN_SPECIALIZATION_INFO_NONE =
    std::make_shared<SpecializationInfo>();

} // namespace merian
