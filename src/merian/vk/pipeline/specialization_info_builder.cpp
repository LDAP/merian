#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

SpecializationInfoBuilder&
SpecializationInfoBuilder::add_entry_id_raw(uint32_t constant_id, uint32_t size, const char* data) {
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

uint32_t SpecializationInfoBuilder::add_entry_raw(uint32_t size, const char* data) {
    uint32_t next_free = 0;
    for (uint32_t i = 0; i <= entries.size(); i++) {
        if (!entries.contains(i)) {
            next_free = i;
        }
    }
    add_entry_id_raw(next_free, size, data);
    return next_free;
}

uint32_t SpecializationInfoBuilder::add_entry(const bool& entry) {
    const VkBool32 entry32 = static_cast<VkBool32>(entry);
    return add_entry(entry32);
}

SpecializationInfoHandle SpecializationInfoBuilder::build() const {
    std::vector<vk::SpecializationMapEntry> vec_entries;
    vec_entries.reserve(entries.size());
    for (const auto& map_entry : entries) {
        vec_entries.push_back(map_entry.second);
    }
    SpecializationInfoHandle result =
        std::make_shared<SpecializationInfo>(vec_entries, data_size, data);
    return result;
}

} // namespace merian
