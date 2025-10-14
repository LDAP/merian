#include "merian/utils/properties_json_load.hpp"

#include <fstream>

using json = nlohmann::json;

namespace merian {

static float decode_float(json& j) {
    if (j.type() == json::value_t::string) {
        std::string encoded = j.template get<std::string>();
        return std::atof(encoded.c_str());
    }
    return j.template get<float>();
}

template <typename T>
static bool load_if_exist(json& j, const std::string& id, T* value, const int components) {
    bool changed = false;
    if (j.contains(id)) {
        json::iterator it = j[id].begin();
        for (int i = 0; i < components; i++, it++) {
            assert(it != j[id].end());
            const T new_value = *it;
            changed |= (new_value != value[i]);
            value[i] = new_value;
        }
    }
    return changed;
}

static bool load_if_exist(json& j, const std::string& id, float* value, const int components) {
    bool changed = false;
    if (j.contains(id)) {
        json::iterator it = j[id].begin();
        for (int i = 0; i < components; i++, it++) {
            assert(it != j[id].end());
            const float new_value = decode_float(*it);
            changed |= (new_value != value[i]);
            value[i] = new_value;
        }
    }
    return changed;
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

JSONLoadProperties::JSONLoadProperties(const nlohmann::json& json) : o(1) {
    o[0] = json;
}

JSONLoadProperties::JSONLoadProperties(const nlohmann::json&& json) : o(1) {
    o[0] = json;
}

JSONLoadProperties::~JSONLoadProperties() {}

bool JSONLoadProperties::st_begin_child(const std::string& id,
                                        const std::string& /*label*/,
                                        const ChildFlags /*flags*/) {
    if (o.back().contains(id)) {
        o.push_back(o.back()[id]);
        return true;
    }
    return false;
}
void JSONLoadProperties::st_end_child() {
    o.pop_back();
}
std::vector<std::string> JSONLoadProperties::st_list_children() {
    std::vector<std::string> children;
    for (auto it = o.back().begin(); it != o.back().end(); it++) {
        if (it.value().is_object()) {
            children.emplace_back(it.key());
        }
    }
    return children;
}

void JSONLoadProperties::st_separate(const std::string& /*label*/) {}
void JSONLoadProperties::st_no_space() {}

void JSONLoadProperties::output_text(const std::string& /*text*/) {}
void JSONLoadProperties::output_plot_line(const std::string& /*label*/,
                                          const float* /*samples*/,
                                          const uint32_t /*count*/,
                                          const float /*scale_min*/,
                                          const float /*scale_max*/) {}

bool JSONLoadProperties::config_float(const std::string& id,
                                      float* value,
                                      const std::string& /*desc*/,
                                      const int components) {
    return load_if_exist(o.back(), id, value, components);
}
bool JSONLoadProperties::config_int(const std::string& id,
                                    int32_t* value,
                                    const std::string& /*desc*/,
                                    const int components) {
    return load_if_exist(o.back(), id, value, components);
}
bool JSONLoadProperties::config_uint(const std::string& id,
                                     uint32_t* value,
                                     const std::string& /*desc*/,
                                     const int components) {
    return load_if_exist(o.back(), id, value, components);
}
bool JSONLoadProperties::config_uint64(const std::string& id,
                                       uint64_t* value,
                                       const std::string& /*desc*/,
                                       const int components) {
    return load_if_exist(o.back(), id, value, components);
}

bool JSONLoadProperties::config_bool(const std::string& id,
                                     bool& value,
                                     const std::string& /*desc*/) {
    const bool old_value = value;
    if (o.back().contains(id))
        value = o.back()[id].template get<bool>();
    return old_value != value;
}
bool JSONLoadProperties::config_bool(const std::string& id, const std::string& /*desc*/) {
    if (o.back().contains(id))
        return o.back()[id].template get<bool>();
    return false;
}
bool JSONLoadProperties::config_options(const std::string& id,
                                        int& selected,
                                        const std::vector<std::string>& options,
                                        const OptionsStyle /*style*/,
                                        const std::string& /*desc*/) {
    const int old_selected = selected;
    if (o.back().contains(id)) {
        std::string option = o.back()[id].template get<std::string>();
        for (int i = 0; i < (int)options.size(); i++) {
            if (options[i] == option)
                selected = i;
        }
    }
    return old_selected != selected;
}
bool JSONLoadProperties::config_text(const std::string& id,
                                     std::string& string,
                                     const bool /*needs_submit*/,
                                     const std::string& /*desc*/) {

    if (o.back().contains(id)) {
        string = o.back()[id].template get<std::string>();
        return true;
    }

    return false;
}
bool JSONLoadProperties::config_text_multiline(const std::string& id,
                                               std::string& string,
                                               const bool /*needs_submit*/,
                                               const std::string& /*desc*/) {

    if (o.back().contains(id)) {
        string = o.back()[id].template get<std::string>();
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
