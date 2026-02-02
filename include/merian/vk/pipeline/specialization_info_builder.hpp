#pragma once

#include "merian/vk/pipeline/specialization_info.hpp"
#include <map>
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

#define MERIAN_INITIAL_SPEC_CONSTANT_ALLOC_SIZE 32

/**
 * @brief      Builds a SpecializationInfo.
 *
 */
class SpecializationInfoBuilder {

  public:
    SpecializationInfoBuilder(const SpecializationInfoBuilder&) = delete;
    SpecializationInfoBuilder(const SpecializationInfoBuilder&&) = delete;

    explicit SpecializationInfoBuilder() {
        data = new char[MERIAN_INITIAL_SPEC_CONSTANT_ALLOC_SIZE];
        alloc_size = MERIAN_INITIAL_SPEC_CONSTANT_ALLOC_SIZE;
    }

    ~SpecializationInfoBuilder() {
        delete[] data;
    }

    // -----------------------------------------------------------------

    SpecializationInfoBuilder&
    add_entry_id_raw(uint32_t constant_id, uint32_t size, const char* data);

    template <typename T> SpecializationInfoBuilder& add_entry_id(uint32_t constant_id, T entry) {
        add_entry_id_raw(constant_id, sizeof(T), reinterpret_cast<char*>(&entry));
        return *this;
    }

    template <typename T>
    SpecializationInfoBuilder& add_entry_id_p(uint32_t constant_id, T* entry) {
        add_entry_id_raw(constant_id, sizeof(T), reinterpret_cast<char*>(entry));
        return *this;
    }

    // Assigns the constant to the next free constant id (use that in your shader). The id is
    // returned.
    uint32_t add_entry_raw(uint32_t size, const char* data);

    uint32_t add_entry(const bool& entry);

    // Assigns the constant to the next free constant id (use that in your shader). The id is
    // returned.
    template <typename T> uint32_t add_entry(const T& entry) {
        return add_entry_raw(sizeof(T), reinterpret_cast<const char*>(&entry));
    }

    // Assigns the constant to the next free constant id (use that in your shader). The id is
    // returned.
    template <typename T> uint32_t add_entry_p(const T* entry) {
        return add_entry_raw(sizeof(T), reinterpret_cast<char*>(entry));
    }

    // Assigns the constants to the next free constant ids (use that in your shader). The ids are
    // returned.
    template <typename... T> std::vector<uint32_t> add_entry(const T&... entries) {
        std::vector<uint32_t> ids = {add_entry(entries)...};
        return ids;
    }

    // -----------------------------------------------------------------

    SpecializationInfoHandle build() const;

  private:
    char* data = nullptr;
    uint32_t alloc_size;
    uint32_t data_size = 0;

    // (constant_id, entry)
    std::map<uint32_t, vk::SpecializationMapEntry> entries;
};

} // namespace merian
