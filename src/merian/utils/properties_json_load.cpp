#include "properties_json_load.hpp"
#include "merian/utils/glm.hpp"

#include <fstream>

using json = nlohmann::json;

namespace merian {

float decode_float(json& j) {
    if (j.type() == json::value_t::string) {
        std::string encoded = j.template get<std::string>();
        return std::atof(encoded.c_str());
    }
    return j.template get<float>();
}

void load_vec3(json& j, glm::vec3& v) {
    json::iterator it = j.begin();
    for (int i = 0; i < 3; i++, it++) {
        v[i] = decode_float(*it);
    }
}

void load_vec4(json& j, glm::vec4& v) {
    json::iterator it = j.begin();
    for (int i = 0; i < 4; i++, it++) {
        v[i] = decode_float(*it);
    }
}

void load_vec3(json& j, glm::uvec3& v) {
    json::iterator it = j.begin();
    for (int i = 0; i < 3; i++, it++) {
        v[i] = *it;
    }
}

void load_vec4(json& j, glm::uvec4& v) {
    json::iterator it = j.begin();
    for (int i = 0; i < 4; i++, it++) {
        v[i] = *it;
    }
}

JSONLoadProperties::JSONLoadProperties(const std::string& json_string) : o(1) {
    o[0] = json::parse(json_string);
}

JSONLoadProperties::JSONLoadProperties(const std::filesystem::path& filename) : o(1) {
    if (std::filesystem::exists(filename)) {
        std::ifstream i(filename.string());
        i >> o[0];
    } else {
        o[0] = json::object();
    }
}

JSONLoadProperties::~JSONLoadProperties() {}

bool JSONLoadProperties::st_begin_child(const std::string& id,
                                        const std::string&,
                                        const ChildFlags) {
    if (o.back().contains(id)) {
        o.push_back(o.back()[id]);
        return true;
    }
    return false;
}
void JSONLoadProperties::st_end_child() {
    o.pop_back();
}

void JSONLoadProperties::st_separate(const std::string&) {}
void JSONLoadProperties::st_no_space() {}

void JSONLoadProperties::output_text(const std::string&) {}
void JSONLoadProperties::output_plot_line(
    const std::string&, const float*, const uint32_t, const float, const float) {}

bool JSONLoadProperties::config_color(const std::string& id, glm::vec3& color, const std::string&) {
    const glm::vec3 old_color = color;
    if (o.back().contains(id))
        load_vec3(o.back()[id], color);
    return old_color != color;
}
bool JSONLoadProperties::config_color(const std::string& id, glm::vec4& color, const std::string&) {
    const glm::vec4 old_color = color;
    if (o.back().contains(id))
        load_vec4(o.back()[id], color);
    return old_color != color;
}
bool JSONLoadProperties::config_vec(const std::string& id, glm::vec3& value, const std::string&) {
    const glm::vec3 old_value = value;
    if (o.back().contains(id))
        load_vec3(o.back()[id], value);
    return old_value != value;
}
bool JSONLoadProperties::config_vec(const std::string& id, glm::vec4& value, const std::string&) {
    const glm::vec4 old_value = value;
    if (o.back().contains(id))
        load_vec4(o.back()[id], value);
    return old_value != value;
}
bool JSONLoadProperties::config_vec(const std::string& id, glm::uvec3& value, const std::string&) {
    const glm::uvec3 old_value = value;
    if (o.back().contains(id))
        load_vec3(o.back()[id], value);
    return old_value != value;
}
bool JSONLoadProperties::config_vec(const std::string& id, glm::uvec4& value, const std::string&) {
    const glm::uvec4 old_value = value;
    if (o.back().contains(id))
        load_vec4(o.back()[id], value);
    return old_value != value;
}
bool JSONLoadProperties::config_angle(
    const std::string& id, float& angle, const std::string&, const float, const float) {
    const float old_angle = angle;
    if (o.back().contains(id))
        angle = decode_float(o.back()[id]);
    return old_angle != angle;
}
bool JSONLoadProperties::config_percent(const std::string& id, float& value, const std::string&) {
    const float old_value = value;
    if (o.back().contains(id))
        value = decode_float(o.back()[id]);
    return old_value != value;
}
bool JSONLoadProperties::config_float(const std::string& id,
                                      float& value,
                                      const std::string&,
                                      const float) {
    const float old_value = value;
    if (o.back().contains(id))
        value = decode_float(o.back()[id]);
    return value != old_value;
}
bool JSONLoadProperties::config_float(
    const std::string& id, float& value, const float&, const float&, const std::string&) {
    const float old_value = value;
    if (o.back().contains(id))
        value = decode_float(o.back()[id]);
    return old_value != value;
}
bool JSONLoadProperties::config_int(const std::string& id, int& value, const std::string&) {
    const int old_value = value;
    if (o.back().contains(id))
        value = o.back()[id].template get<int>();
    return old_value != value;
}
bool JSONLoadProperties::config_int(
    const std::string& id, int& value, const int&, const int&, const std::string&) {
    const int old_value = value;
    if (o.back().contains(id))
        value = o.back()[id].template get<int>();
    return old_value != value;
}
bool JSONLoadProperties::config_uint(const std::string& id, uint32_t& value, const std::string&) {
    const uint32_t old_value = value;
    if (o.back().contains(id))
        value = o.back()[id].template get<uint32_t>();
    return old_value != value;
}
bool JSONLoadProperties::config_uint(
    const std::string& id, uint32_t& value, const uint32_t&, const uint32_t&, const std::string&) {
    const uint32_t old_value = value;
    if (o.back().contains(id))
        value = o.back()[id].template get<uint32_t>();
    return old_value != value;
}
bool JSONLoadProperties::config_float3(const std::string& id, float value[3], const std::string&) {
    const float old_value[3] = {value[0], value[1], value[2]};
    if (o.back().contains(id))
        load_vec3(o.back()[id], *merian::as_vec3(value));
    return old_value[0] != value[0] || old_value[1] != value[1] || old_value[2] != value[2];
}
bool JSONLoadProperties::config_bool(const std::string& id, bool& value, const std::string&) {
    const bool old_value = value;
    if (o.back().contains(id))
        value = o.back()[id].template get<bool>();
    return old_value != value;
}
bool JSONLoadProperties::config_bool(const std::string& id, const std::string&) {
    if (o.back().contains(id))
        return o.back()[id].template get<bool>();
    return false;
}
bool JSONLoadProperties::config_options(const std::string& id,
                                        int& selected,
                                        const std::vector<std::string>& options,
                                        const OptionsStyle,
                                        const std::string&) {
    const int old_selected = selected;
    if (o.back().contains(id)) {
        std::string option = o.back()[id].template get<std::string>();
        for (uint32_t i = 0; i < options.size(); i++) {
            if (options[i] == option)
                selected = i;
        }
    }
    return old_selected != selected;
}
bool JSONLoadProperties::config_text(const std::string& id,
                                     [[maybe_unused]] const uint32_t max_len,
                                     char* string,
                                     const bool,
                                     const std::string&) {

    if (o.back().contains(id)) {
        std::string s = o.back()[id].template get<std::string>();
        assert(s.size() < max_len);
        strcpy(string, s.c_str());
        return true;
    }

    return false;
}
bool JSONLoadProperties::config_text_multiline(const std::string& id,
                                               [[maybe_unused]] const uint32_t max_len,
                                               char* string,
                                               const bool,
                                               const std::string&) {

    if (o.back().contains(id)) {
        std::string s = o.back()[id].template get<std::string>();
        assert(s.size() < max_len);
        strcpy(string, s.c_str());
        return true;
    }

    return false;
}

bool JSONLoadProperties::is_ui() {
    return false;
}
bool JSONLoadProperties::serialize_json(const std::string& id, nlohmann::json& json) {
    if (o.back().contains(id)) {
        json = o.back()[id];
        return true;
    }

    return false;
}
bool JSONLoadProperties::serialize_string(const std::string& id, std::string& s) {
    if (o.back().contains(id)) {
        s = o.back()[id].template get<std::string>();
        return true;
    }

    return false;
}

} // namespace merian
