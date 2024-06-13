#include "configuration_json_dump.hpp"
#include "merian/utils/glm.hpp"

#include <cmath>
#include <fstream>

using json = nlohmann::json;

namespace merian {

json encode_float(float f) {
    if (std::isinf(f) || std::isnan(f))
        return std::to_string(f);
    else
        return f;
}

json dump_vec3(glm::vec3& v) {
    json j;
    for (int i = 0; i < 3; i++) {
        j.push_back(encode_float(v[i]));
    }
    return j;
}

json dump_vec4(glm::vec4& v) {
    json j;
    for (int i = 0; i < 4; i++) {
        j.push_back(encode_float(v[i]));
    }
    return j;
}

JSONDumpConfiguration::JSONDumpConfiguration(const std::optional<std::filesystem::path>& filename)
    : filename(filename), o(1) {}

JSONDumpConfiguration::~JSONDumpConfiguration() {
    assert(o.size() == 1 && "Missing st_end_child?");

    if (filename) {
        std::ofstream file(filename.value().string());
        file << std::setw(4) << current() << std::endl;
    }
}

bool JSONDumpConfiguration::st_begin_child(const std::string& id, const std::string&) {
    o.emplace_back(id, nlohmann::json());
    return true;
}
void JSONDumpConfiguration::st_end_child() {
    if (!current().empty())
        o[o.size() - 2].second[o.back().first] = current();
    o.pop_back();
}

bool JSONDumpConfiguration::st_new_section(const std::string&) {
    return true;
}
void JSONDumpConfiguration::st_separate(const std::string&) {}
void JSONDumpConfiguration::st_no_space() {}

void JSONDumpConfiguration::output_text(const std::string&) {}
void JSONDumpConfiguration::output_plot_line(
    const std::string&, const float*, const uint32_t, const float, const float) {}

void JSONDumpConfiguration::config_color(const std::string& id,
                                         glm::vec3& color,
                                         const std::string&) {
    current()[id] = dump_vec3(color);
}
void JSONDumpConfiguration::config_color(const std::string& id,
                                         glm::vec4& color,
                                         const std::string&) {
    current()[id] = dump_vec4(color);
}
void JSONDumpConfiguration::config_vec(const std::string& id,
                                       glm::vec3& value,
                                       const std::string&) {
    current()[id] = dump_vec3(value);
}
void JSONDumpConfiguration::config_vec(const std::string& id,
                                       glm::vec4& value,
                                       const std::string&) {
    current()[id] = dump_vec4(value);
}
void JSONDumpConfiguration::config_angle(
    const std::string& id, float& angle, const std::string&, const float, const float) {
    current()[id] = encode_float(angle);
}
void JSONDumpConfiguration::config_percent(const std::string& id,
                                           float& value,
                                           const std::string&) {
    current()[id] = encode_float(value);
}
void JSONDumpConfiguration::config_float(const std::string& id,
                                         float& value,
                                         const std::string&,
                                         const float) {
    current()[id] = encode_float(value);
}
void JSONDumpConfiguration::config_float(
    const std::string& id, float& value, const float&, const float&, const std::string&) {
    current()[id] = encode_float(value);
}
void JSONDumpConfiguration::config_int(const std::string& id, int& value, const std::string&) {
    current()[id] = value;
}
void JSONDumpConfiguration::config_int(
    const std::string& id, int& value, const int&, const int&, const std::string&) {
    current()[id] = value;
}
void JSONDumpConfiguration::config_uint(const std::string& id,
                                        uint32_t& value,
                                        const std::string&) {
    current()[id] = value;
}
void JSONDumpConfiguration::config_uint(
    const std::string& id, uint32_t& value, const uint32_t&, const uint32_t&, const std::string&) {
    current()[id] = value;
}
void JSONDumpConfiguration::config_float3(const std::string& id,
                                          float value[3],
                                          const std::string&) {
    current()[id] = dump_vec3(*merian::as_vec3(value));
}
void JSONDumpConfiguration::config_bool(const std::string& id, bool& value, const std::string&) {
    current()[id] = value;
}
bool JSONDumpConfiguration::config_bool(const std::string&, const std::string&) {
    return false;
}
void JSONDumpConfiguration::config_options(const std::string& id,
                                           int& selected,
                                           const std::vector<std::string>& options,
                                           const OptionsStyle,
                                           const std::string&) {
    current()[id] = options[selected];
}
bool JSONDumpConfiguration::config_text(
    const std::string& id, const uint32_t, char* buf, const bool needs_submit, const std::string&) {
    if (!needs_submit) {
        current()[id] = buf;
    }
    return false;
}
bool JSONDumpConfiguration::config_text_multiline(
    const std::string& id, const uint32_t, char* buf, const bool, const std::string&) {
    current()[id] = buf;
    return false;
}

} // namespace merian
