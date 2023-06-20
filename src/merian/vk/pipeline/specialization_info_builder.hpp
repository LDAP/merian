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
    SpecializationInfoBuilder() {
        data = new char[MERIAN_INITIAL_SPEC_CONSTANT_ALLOC_SIZE];
        alloc_size = MERIAN_INITIAL_SPEC_CONSTANT_ALLOC_SIZE;
    }

    ~SpecializationInfoBuilder() {
        delete[] data;
    }

    // -----------------------------------------------------------------

    SpecializationInfoBuilder& add_entry_id_raw(uint32_t constant_id, uint32_t size, char* data) {
        assert(!entries.contains(constant_id));

        while (size + data_size > alloc_size) {
            // double size
            uint32_t new_alloc_size = alloc_size * 2;
            char* new_data = new char[new_alloc_size];

            memcpy(new_data, this->data, data_size);
            delete[] this->data;

            this->data = new_data;
            this->alloc_size = new_alloc_size;
        }

        uint32_t offset = data_size;
        data_size += size;
        memcpy(this->data + offset, data, size);
        entries[constant_id] = vk::SpecializationMapEntry{constant_id, offset, size};
        return *this;
    }

    template<typename T>
    SpecializationInfoBuilder& add_entry_id(uint32_t constant_id, T entry) {
        add_entry_id_raw(constant_id, sizeof(T), reinterpret_cast<char*>(&entry));
        return *this;
    }

    template<typename T>
    SpecializationInfoBuilder& add_entry_id_p(uint32_t constant_id, T* entry) {
        add_entry_id_raw(constant_id, sizeof(T), reinterpret_cast<char*>(entry));
        return *this;
    }

    // Assigns the constant to the next free constant id (use that in your shader). The id is
    // returned.
    uint32_t add_entry_raw(uint32_t size, char* data) {
        uint32_t next_free = 0;
        for (uint32_t i = 0; i <= entries.size(); i++) {
            if (!entries.contains(i)) {
                next_free = i;
            }
        }
        add_entry_id_raw(next_free, size, data);
        return next_free;
    }

    // Assigns the constant to the next free constant id (use that in your shader). The id is
    // returned.
    template<typename T>
    uint32_t add_entry(T entry) {
        return add_entry_raw(sizeof(T), reinterpret_cast<char*>(&entry));
    }

    // Assigns the constant to the next free constant id (use that in your shader). The id is
    // returned.
    template<typename T>
    uint32_t add_entry_p(T* entry) {
        return add_entry_raw(sizeof(T), reinterpret_cast<char*>(entry));
    }

    // Assigns the constants to the next free constant ids (use that in your shader). The ids are
    // returned.
    template<typename... T>
    std::vector<uint32_t> add_entry(T... entries) {
        std::vector<uint32_t> ids = {add_entry(entries)...};
        return ids;
    }

    // -----------------------------------------------------------------

    SpecializationInfoHandle build() const {
        std::vector<vk::SpecializationMapEntry> vec_entries;
        for (auto& map_entry : entries) {
            vec_entries.push_back(map_entry.second);
        }
        SpecializationInfoHandle result = std::make_shared<SpecializationInfo>(vec_entries, data_size, data);
        return result;
    }

  private:
    char* data = nullptr;
    uint32_t alloc_size;
    uint32_t data_size = 0;

    // (constant_id, entry)
    std::map<uint32_t, vk::SpecializationMapEntry> entries;
};

} // namespace merian
