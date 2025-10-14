#include "merian/utils/properties_json_dump.hpp"

#include <cmath>
#include <fstream>

using json = nlohmann::json;

namespace merian {

static json encode_float(float f) {
    if (std::isinf(f) || std::isnan(f))
        return std::to_string(f);
    return f;
}

template <typename T> static json dump(T* v, const int components) {
    json j;
    for (int i = 0; i < components; i++) {
        j.push_back(v[i]);
    }
    return j;
}

static json dump(float* v, const int components) {
    json j;
    for (int i = 0; i < components; i++) {
        j.push_back(encode_float(v[i]));
    }
    return j;
}

JSONDumpProperties::JSONDumpProperties(const std::optional<std::filesystem::path>& filename)
    : filename(filename), o(1) {}

JSONDumpProperties::~JSONDumpProperties() {
    assert(o.size() == 1 && "Missing st_end_child?");

    if (filename) {
        std::ofstream file(filename.value().string());
        file << std::setw(4) << current() << '\n';
    }
}

bool JSONDumpProperties::st_begin_child(const std::string& id,
                                        const std::string& /*label*/,
                                        const ChildFlags /*flags*/) {
    o.emplace_back(id, nlohmann::json());
    return true;
}
void JSONDumpProperties::st_end_child() {
    if (!current().empty())
        o[o.size() - 2].second[o.back().first] = current();
    o.pop_back();
}

void JSONDumpProperties::st_separate(const std::string& /*label*/) {}
void JSONDumpProperties::st_no_space() {}

void JSONDumpProperties::output_text(const std::string& /*text*/) {}
void JSONDumpProperties::output_plot_line(const std::string& /*label*/,
                                          const float* /*samples*/,
                                          const uint32_t /*count*/,
                                          const float /*scale_min*/,
                                          const float /*scale_max*/) {}

bool JSONDumpProperties::config_float(const std::string& id,
                                      float* value,
                                      const std::string& /*desc*/,
                                      const int components) {
    current()[id] = dump(value, components);
    return false;
}
bool JSONDumpProperties::config_int(const std::string& id,
                                    int32_t* value,
                                    const std::string& /*desc*/,
                                    const int components) {
    current()[id] = dump(value, components);
    return false;
}
bool JSONDumpProperties::config_uint(const std::string& id,
                                     uint32_t* value,
                                     const std::string& /*desc*/,
                                     const int components) {
    current()[id] = dump(value, components);
    return false;
}
bool JSONDumpProperties::config_uint64(const std::string& id,
                                       uint64_t* value,
                                       const std::string& /*desc*/,
                                       const int components) {
    current()[id] = dump(value, components);
    return false;
}

bool JSONDumpProperties::config_bool(const std::string& id,
                                     bool& value,
                                     const std::string& /*desc*/) {
    current()[id] = value;
    return false;
}
bool JSONDumpProperties::config_bool(const std::string& /*id*/, const std::string& /*desc*/) {
    return false;
}
bool JSONDumpProperties::config_options(const std::string& id,
                                        int& selected,
                                        const std::vector<std::string>& options,
                                        const OptionsStyle /*style*/,
                                        const std::string& /*desc*/) {
    if (selected >= static_cast<int>(options.size())) {
        return false;
    }
    current()[id] = options[selected];
    return false;
}
bool JSONDumpProperties::config_text(const std::string& id,
                                     std::string& string,
                                     const bool /*needs_submit*/,
                                     const std::string& /*desc*/) {
    current()[id] = string;
    return false;
}
bool JSONDumpProperties::config_text_multiline(const std::string& id,
                                               std::string& string,
                                               const bool /*needs_submit*/,
                                               const std::string& /*desc*/) {
    current()[id] = string;
    return false;
}

bool JSONDumpProperties::is_ui() {
    return false;
}
bool JSONDumpProperties::serialize_json(const std::string& id, nlohmann::json& json) {
    current()[id] = json;
    return false;
}
bool JSONDumpProperties::serialize_string(const std::string& id, std::string& s) {
    current()[id] = s;
    return false;
}

} // namespace merian
