#include "configuration_json_dump.hpp"
#include "merian/utils/glm.hpp"

#include <cmath>
#include <fstream>

using json = nlohmann::json;

namespace merian {

json encode_float(float f) {
    if (std::isinf(f) || std::isnan(f))
        return std::to_string(f);
    else return f;
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

JSONDumpConfiguration::JSONDumpConfiguration(const std::string& filename)
    : filename(filename), o(1) {}
JSONDumpConfiguration::~JSONDumpConfiguration() {
    std::ofstream file(filename);
    file << std::setw(4) << o[0] << std::endl;
}

bool JSONDumpConfiguration::st_begin_child(const std::string& id, const std::string&) {
    object_name = id;
    o.emplace_back();
    return true;
}
void JSONDumpConfiguration::st_end_child() {
    o[o.size() - 2][object_name] = o[o.size() - 1];
    o.pop_back();
}

bool JSONDumpConfiguration::st_new_section(const std::string&) {
    return true;
}
void JSONDumpConfiguration::st_separate(const std::string&) {}
void JSONDumpConfiguration::st_no_space() {}

void JSONDumpConfiguration::output_text(const std::string&) {}
void JSONDumpConfiguration::output_plot_line(const std::string&,
                                             const std::vector<float>&,
                                             const float,
                                             const float) {}

void JSONDumpConfiguration::config_color(const std::string& id,
                                         glm::vec3& color,
                                         const std::string&) {
    o.back()[id] = dump_vec3(color);
}
void JSONDumpConfiguration::config_color(const std::string& id,
                                         glm::vec4& color,
                                         const std::string&) {
    o.back()[id] = dump_vec4(color);
}
void JSONDumpConfiguration::config_vec(const std::string& id,
                                       glm::vec3& value,
                                       const std::string&) {
    o.back()[id] = dump_vec3(value);
}
void JSONDumpConfiguration::config_vec(const std::string& id,
                                       glm::vec4& value,
                                       const std::string&) {
    o.back()[id] = dump_vec4(value);
}
void JSONDumpConfiguration::config_angle(
    const std::string& id, float& angle, const std::string&, const float, const float) {
    o.back()[id] = encode_float(angle);
}
void JSONDumpConfiguration::config_percent(const std::string& id,
                                           float& value,
                                           const std::string&) {
    o.back()[id] = encode_float(value);
}
void JSONDumpConfiguration::config_float(const std::string& id,
                                         float& value,
                                         const std::string&,
                                         const float) {
    o.back()[id] = encode_float(value);
}
void JSONDumpConfiguration::config_float(
    const std::string& id, float& value, const float&, const float&, const std::string&) {
    o.back()[id] = encode_float(value);
}
void JSONDumpConfiguration::config_int(const std::string& id, int& value, const std::string&) {
    o.back()[id] = value;
}
void JSONDumpConfiguration::config_int(
    const std::string& id, int& value, const int&, const int&, const std::string&) {
    o.back()[id] = value;
}
void JSONDumpConfiguration::config_float3(const std::string& id,
                                          float value[3],
                                          const std::string&) {
    o.back()[id] = dump_vec3(*merian::as_vec3(value));
}
void JSONDumpConfiguration::config_bool(const std::string& id, bool& value, const std::string&) {
    o.back()[id] = value;
}
bool JSONDumpConfiguration::config_bool(const std::string&, const std::string&) {
    return false;
}
void JSONDumpConfiguration::config_options(const std::string& id,
                                           int& selected,
                                           const std::vector<std::string>& options,
                                           const OptionsStyle,
                                           const std::string&) {
    o.back()[id] = options[selected];
}
bool JSONDumpConfiguration::config_text(
    const std::string&, const uint32_t, char*, const bool, const std::string&) {
    return false;
}

} // namespace merian
