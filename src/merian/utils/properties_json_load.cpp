#include "merian/utils/properties_json_load.hpp"

#include <algorithm>
#include <fmt/ranges.h>
#include <fstream>
#include <iterator>
#include <ranges>
#include <stdexcept>

using json = nlohmann::json;

namespace merian {

static float decode_float(const json& j) {
    if (j.type() == json::value_t::string) {
        std::string encoded = j.template get<std::string>();
        return std::atof(encoded.c_str());
    }
    return j.template get<float>();
}

template <typename T, typename Decode>
static bool
load(const json& j, const std::string& id, T* value, const int components, const Decode& decode) {
    const json& field = j.at(id);
    if (field.is_array() ? field.size() < static_cast<std::size_t>(components) : components != 1) {
        throw std::invalid_argument{fmt::format("'{}' needs {} values", id, components)};
    }
    bool changed = false;
    for (int i = 0; i < components; i++) {
        const T new_value = decode(field.is_array() ? field.at(i) : field);
        changed |= new_value != value[i];
        value[i] = new_value;
    }
    return changed;
}

template <typename T> static T decode(const json& j) {
    return j.template get<T>();
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
                                        const ChildFlags flags) {
    // a TABS child groups for a UI only, so it is not a level here
    if ((flags & ChildFlagBits::TABS) != 0) {
        transparent.push_back(true);
        return true;
    }
    if (read(id)) {
        transparent.push_back(false);
        o.push_back(o.back()[id]);
        current /= id;
        entered_children.insert(current.to_string());
        return true;
    }
    return false;
}
void JSONLoadProperties::st_end_child() {
    const bool was_transparent = transparent.back();
    transparent.pop_back();
    if (!was_transparent) {
        o.pop_back();
        current = current.parent_pointer();
    }
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
                                      const int components,
                                      const float /*sensitivity*/,
                                      const std::optional<float>& /*min*/,
                                      const std::optional<float>& /*max*/) {
    return read(id) && load(o.back(), id, value, components, decode_float);
}
bool JSONLoadProperties::config_int(const std::string& id,
                                    int32_t* value,
                                    const std::string& /*desc*/,
                                    const int components,
                                    const std::optional<int32_t>& /*min*/,
                                    const std::optional<int32_t>& /*max*/) {
    return read(id) && load(o.back(), id, value, components, decode<int32_t>);
}
bool JSONLoadProperties::config_uint(const std::string& id,
                                     uint32_t* value,
                                     const std::string& /*desc*/,
                                     const int components,
                                     const std::optional<uint32_t>& /*min*/,
                                     const std::optional<uint32_t>& /*max*/) {
    return read(id) && load(o.back(), id, value, components, decode<uint32_t>);
}
bool JSONLoadProperties::config_uint64(const std::string& id,
                                       uint64_t* value,
                                       const std::string& /*desc*/,
                                       const int components,
                                       const std::optional<uint64_t>& /*min*/,
                                       const std::optional<uint64_t>& /*max*/) {
    return read(id) && load(o.back(), id, value, components, decode<uint64_t>);
}

bool JSONLoadProperties::config_bool(const std::string& id,
                                     bool& value,
                                     const std::string& /*desc*/) {
    const bool old_value = value;
    if (read(id)) {
        value = o.back()[id].template get<bool>();
    }
    return old_value != value;
}
bool JSONLoadProperties::config_bool(const std::string& id, const std::string& /*desc*/) {
    if (read(id)) {
        return o.back()[id].template get<bool>();
    }
    return false;
}
bool JSONLoadProperties::config_options(const std::string& id,
                                        int& selected,
                                        const std::vector<std::string>& options,
                                        const OptionsStyle /*style*/,
                                        const std::string& /*desc*/) {
    const int old_selected = selected;
    if (read(id)) {
        const std::string option = o.back()[id].template get<std::string>();
        const auto it = std::ranges::find(options, option);
        if (it == options.end()) {
            constexpr std::size_t MAX_LISTED = 10;
            const auto listed = options | std::views::take(MAX_LISTED);
            invalid.push_back(fmt::format("{}: '{}' is none of {}{}", (current / id).to_string(),
                                          option, fmt::join(listed, ", "),
                                          options.size() > MAX_LISTED ? ", ..." : ""));
        } else {
            selected = static_cast<int>(std::distance(options.begin(), it));
        }
    }
    return old_selected != selected;
}
bool JSONLoadProperties::config_text(const std::string& id,
                                     std::string& string,
                                     const bool /*needs_submit*/,
                                     const std::string& /*desc*/) {

    if (read(id)) {
        string = o.back()[id].template get<std::string>();
        return true;
    }

    return false;
}
bool JSONLoadProperties::config_text_multiline(const std::string& id,
                                               std::string& string,
                                               const bool /*needs_submit*/,
                                               const std::string& /*desc*/) {

    if (read(id)) {
        string = o.back()[id].template get<std::string>();
        return true;
    }

    return false;
}

bool JSONLoadProperties::is_ui() {
    return false;
}
bool JSONLoadProperties::serialize_json(const std::string& id, nlohmann::json& json) {
    if (read(id)) {
        json = o.back()[id];
        return true;
    }

    return false;
}
bool JSONLoadProperties::serialize_string(const std::string& id, std::string& s) {
    if (read(id)) {
        s = o.back()[id].template get<std::string>();
        return true;
    }

    return false;
}

std::vector<std::string> JSONLoadProperties::unread_keys() const {
    std::vector<std::string> unread;
    collect_unread(o.front(), json::json_pointer(), unread);
    return unread;
}

const std::vector<std::string>& JSONLoadProperties::invalid_values() const {
    return invalid;
}

std::vector<std::string> JSONLoadProperties::problems() const {
    std::vector<std::string> problems = invalid;
    if (const std::vector<std::string> unread = unread_keys(); !unread.empty()) {
        problems.push_back(fmt::format("unknown {}", fmt::join(unread, ", ")));
    }
    return problems;
}

std::vector<std::string>
JSONLoadProperties::load_revealed(const nlohmann::json& json,
                                  const std::function<bool(Properties&)>& load) {
    std::vector<std::string> problems;
    try {
        if (!json.is_null() && !json.is_object()) {
            throw std::invalid_argument{
                fmt::format("properties must be an object, not {}", json.type_name())};
        }
        JSONLoadProperties props{json};
        const bool reads_on = load(props);
        problems = props.invalid_values();
        const std::vector<std::string> unread = props.unread_keys();
        if (reads_on && !unread.empty()) {
            nlohmann::json remaining = nlohmann::json::object();
            for (const std::string& key : unread) {
                const nlohmann::json::json_pointer pointer{key};
                remaining[pointer] = json.at(pointer);
            }
            JSONLoadProperties remaining_props{remaining};
            load(remaining_props);
            std::ranges::copy(remaining_props.problems(), std::back_inserter(problems));
        }
    } catch (const nlohmann::json::exception& e) {
        problems.emplace_back(e.what());
    } catch (const std::invalid_argument& e) {
        problems.emplace_back(e.what());
    }
    return problems;
}

bool JSONLoadProperties::read(const std::string& id) {
    if (!o.back().contains(id)) {
        return false;
    }
    read_keys.insert((current / id).to_string());
    return true;
}

void JSONLoadProperties::collect_unread(const json& object,
                                        const json::json_pointer& pointer,
                                        std::vector<std::string>& unread) const {
    if (!object.is_object()) {
        return;
    }
    for (const auto& [key, value] : object.items()) {
        const json::json_pointer key_pointer = pointer / key;
        const std::string key_string = key_pointer.to_string();
        if (!read_keys.contains(key_string)) {
            unread.push_back(key_string);
        } else if (entered_children.contains(key_string)) {
            collect_unread(value, key_pointer, unread);
        }
    }
}

} // namespace merian
