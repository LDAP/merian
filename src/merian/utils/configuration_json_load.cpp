#include "configuration_json_load.hpp"
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

glm::vec3 load_vec3(json& j) {
    glm::vec3 v;
    json::iterator it = j.begin();
    for (int i = 0; i < 3; i++, it++) {
        v[i] = decode_float(*it);
    }
    return v;
}

glm::vec4 load_vec4(json& j) {
    glm::vec4 v;
    json::iterator it = j.begin();
    for (int i = 0; i < 4; i++, it++) {
        v[i] = decode_float(*it);
    }
    return v;
}

JSONLoadConfiguration::JSONLoadConfiguration(const std::string& filename) : o(1) {
    if (std::filesystem::exists(filename)) {
        std::ifstream i(filename);
        i >> o[0];
    } else {
        o[0] = json::object();
    }
}
JSONLoadConfiguration::~JSONLoadConfiguration() {}

bool JSONLoadConfiguration::st_begin_child(const std::string& id, const std::string&) {
    if (o.back().contains(id)) {
        o.push_back(o.back()[id]);
        return true;
    }
    return false;
}
void JSONLoadConfiguration::st_end_child() {
    o.pop_back();
}

bool JSONLoadConfiguration::st_new_section(const std::string&) {
    return true;
}
void JSONLoadConfiguration::st_separate(const std::string&) {}
void JSONLoadConfiguration::st_no_space() {}

void JSONLoadConfiguration::output_text(const std::string&) {}
void JSONLoadConfiguration::output_plot_line(const std::string&,
                                             const std::vector<float>&,
                                             const float,
                                             const float) {}

void JSONLoadConfiguration::config_color(const std::string& id,
                                         glm::vec3& color,
                                         const std::string&) {
    if (o.back().contains(id))
        color = load_vec3(o.back()[id]);
}
void JSONLoadConfiguration::config_color(const std::string& id,
                                         glm::vec4& color,
                                         const std::string&) {
    if (o.back().contains(id))
        color = load_vec4(o.back()[id]);
}
void JSONLoadConfiguration::config_vec(const std::string& id,
                                       glm::vec3& value,
                                       const std::string&) {
    if (o.back().contains(id))
        value = load_vec3(o.back()[id]);
}
void JSONLoadConfiguration::config_vec(const std::string& id,
                                       glm::vec4& value,
                                       const std::string&) {
    if (o.back().contains(id))
        value = load_vec4(o.back()[id]);
}
void JSONLoadConfiguration::config_angle(
    const std::string& id, float& angle, const std::string&, const float, const float) {
    if (o.back().contains(id))
        angle = decode_float(o.back()[id]);
}
void JSONLoadConfiguration::config_percent(const std::string& id,
                                           float& value,
                                           const std::string&) {
    if (o.back().contains(id))
        value = decode_float(o.back()[id]);
}
void JSONLoadConfiguration::config_float(const std::string& id,
                                         float& value,
                                         const std::string&,
                                         const float) {
    if (o.back().contains(id))
        value = decode_float(o.back()[id]);
}
void JSONLoadConfiguration::config_float(
    const std::string& id, float& value, const float&, const float&, const std::string&) {
    if (o.back().contains(id))
        value = decode_float(o.back()[id]);
}
void JSONLoadConfiguration::config_int(const std::string& id, int& value, const std::string&) {
    if (o.back().contains(id))
        value = o.back()[id].template get<int>();
}
void JSONLoadConfiguration::config_int(
    const std::string& id, int& value, const int&, const int&, const std::string&) {
    if (o.back().contains(id))
        value = o.back()[id].template get<int>();
}
void JSONLoadConfiguration::config_float3(const std::string& id,
                                          float value[3],
                                          const std::string&) {
    if (o.back().contains(id))
        *merian::as_vec3(value) = load_vec3(o.back()[id]);
}
void JSONLoadConfiguration::config_bool(const std::string& id, bool& value, const std::string&) {
    if (o.back().contains(id))
        value = o.back()[id].template get<bool>();
}
bool JSONLoadConfiguration::config_bool(const std::string& id, const std::string&) {
    if (o.back().contains(id))
        return o.back()[id].template get<bool>();
    return false;
}
void JSONLoadConfiguration::config_options(const std::string& id,
                                           int& selected,
                                           const std::vector<std::string>& options,
                                           const OptionsStyle,
                                           const std::string&) {
    if (o.back().contains(id)) {
        std::string option = o.back()[id].template get<std::string>();
        for (uint32_t i = 0; i < options.size(); i++) {
            if (options[i] == option)
                selected = i;
        }
    }
}
bool JSONLoadConfiguration::config_text(
    const std::string& id, const uint32_t max_len, char* string, const bool, const std::string&) {

    if (o.back().contains(id)) {
        std::string s = o.back()[id].template get<std::string>();
        assert(s.size() < max_len);
        strcpy(string, s.c_str());
        return true;
    }

    return false;
}
bool JSONLoadConfiguration::config_text_multiline(
    const std::string& id, const uint32_t max_len, char* string, const bool, const std::string&) {

    if (o.back().contains(id)) {
        std::string s = o.back()[id].template get<std::string>();
        assert(s.size() < max_len);
        strcpy(string, s.c_str());
        return true;
    }

    return false;
}

} // namespace merian
