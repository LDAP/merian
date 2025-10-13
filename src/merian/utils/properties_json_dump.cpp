#include "merian/utils/properties_json_dump.hpp"
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

json dump_vec3(float3& v) {
    json j;
    for (int i = 0; i < 3; i++) {
        j.push_back(encode_float(v[i]));
    }
    return j;
}

json dump_vec4(float4& v) {
    json j;
    for (int i = 0; i < 4; i++) {
        j.push_back(encode_float(v[i]));
    }
    return j;
}

json dump_vec3(uint3& v) {
    json j;
    for (int i = 0; i < 3; i++) {
        j.push_back(v[i]);
    }
    return j;
}

json dump_vec4(uint4& v) {
    json j;
    for (int i = 0; i < 4; i++) {
        j.push_back(v[i]);
    }
    return j;
}

JSONDumpProperties::JSONDumpProperties(const std::optional<std::filesystem::path>& filename)
    : filename(filename), o(1) {}

JSONDumpProperties::~JSONDumpProperties() {
    assert(o.size() == 1 && "Missing st_end_child?");

    if (filename) {
        std::ofstream file(filename.value().string());
        file << std::setw(4) << current() << std::endl;
    }
}

bool JSONDumpProperties::st_begin_child(const std::string& id,
                                        const std::string&,
                                        const ChildFlags) {
    o.emplace_back(id, nlohmann::json());
    return true;
}
void JSONDumpProperties::st_end_child() {
    if (!current().empty())
        o[o.size() - 2].second[o.back().first] = current();
    o.pop_back();
}

void JSONDumpProperties::st_separate(const std::string&) {}
void JSONDumpProperties::st_no_space() {}

void JSONDumpProperties::output_text(const std::string&) {}
void JSONDumpProperties::output_plot_line(
    const std::string&, const float*, const uint32_t, const float, const float) {}

bool JSONDumpProperties::config_color(const std::string& id, float3& color, const std::string&) {
    current()[id] = dump_vec3(color);
    return false;
}
bool JSONDumpProperties::config_color(const std::string& id, float4& color, const std::string&) {
    current()[id] = dump_vec4(color);
    return false;
}
bool JSONDumpProperties::config_vec(const std::string& id, float3& value, const std::string&) {
    current()[id] = dump_vec3(value);
    return false;
}
bool JSONDumpProperties::config_vec(const std::string& id, float4& value, const std::string&) {
    current()[id] = dump_vec4(value);
    return false;
}
bool JSONDumpProperties::config_vec(const std::string& id, uint3& value, const std::string&) {
    current()[id] = dump_vec3(value);
    return false;
}
bool JSONDumpProperties::config_vec(const std::string& id, uint4& value, const std::string&) {
    current()[id] = dump_vec4(value);
    return false;
}
bool JSONDumpProperties::config_angle(
    const std::string& id, float& angle, const std::string&, const float, const float) {
    current()[id] = encode_float(angle);
    return false;
}
bool JSONDumpProperties::config_percent(const std::string& id, float& value, const std::string&) {
    current()[id] = encode_float(value);
    return false;
}
bool JSONDumpProperties::config_float(const std::string& id,
                                      float& value,
                                      const std::string&,
                                      const float) {
    current()[id] = encode_float(value);
    return false;
}
bool JSONDumpProperties::config_float(
    const std::string& id, float& value, const float&, const float&, const std::string&) {
    current()[id] = encode_float(value);
    return false;
}
bool JSONDumpProperties::config_int(const std::string& id, int& value, const std::string&) {
    current()[id] = value;
    return false;
}
bool JSONDumpProperties::config_int(
    const std::string& id, int& value, const int&, const int&, const std::string&) {
    current()[id] = value;
    return false;
}
bool JSONDumpProperties::config_uint(const std::string& id, uint32_t& value, const std::string&) {
    current()[id] = value;
    return false;
}
bool JSONDumpProperties::config_uint(
    const std::string& id, uint32_t& value, const uint32_t&, const uint32_t&, const std::string&) {
    current()[id] = value;
    return false;
}
bool JSONDumpProperties::config_uint(const std::string& id, uint64_t& value, const std::string&) {
    current()[id] = value;
    return false;
}
bool JSONDumpProperties::config_uint(
    const std::string& id, uint64_t& value, const uint64_t&, const uint64_t&, const std::string&) {
    current()[id] = value;
    return false;
}
bool JSONDumpProperties::config_float3(const std::string& id, float value[3], const std::string&) {
    float3 v = load_float3(value);
    current()[id] = dump_vec3(v);
    return false;
}
bool JSONDumpProperties::config_bool(const std::string& id, bool& value, const std::string&) {
    current()[id] = value;
    return false;
}
bool JSONDumpProperties::config_bool(const std::string&, const std::string&) {
    return false;
}
bool JSONDumpProperties::config_options(const std::string& id,
                                        int& selected,
                                        const std::vector<std::string>& options,
                                        const OptionsStyle,
                                        const std::string&) {
    if (selected >= static_cast<int>(options.size())) {
        return false;
    }
    current()[id] = options[selected];
    return false;
}
bool JSONDumpProperties::config_text(const std::string& id,
                                     std::string& string,
                                     const bool,
                                     const std::string&) {
    current()[id] = string;
    return false;
}
bool JSONDumpProperties::config_text_multiline(const std::string& id,
                                               std::string& string,
                                               const bool,
                                               const std::string&) {
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
