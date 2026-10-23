#include "python_properties.hpp"

#include <algorithm>
#include <limits>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <numbers>

namespace nb = nanobind;
using namespace nb::literals;

namespace merian::python {

namespace {

template <typename T> nb::object to_python(const T* value, const int components) {
    if (components == 1) {
        return nb::cast(value[0]);
    }
    nb::list values;
    for (int i = 0; i < components; i++) {
        values.append(value[i]);
    }
    return nb::tuple(values);
}

nb::object bound_to_python(const std::optional<float>& bound) {
    return bound ? nb::cast(*bound) : nb::none();
}

constexpr float DEGREES_PER_RADIAN = 180.f / std::numbers::pi_v<float>;

template <typename T> auto clamp_to(const T min, const T max) {
    return [min, max](const T value) { return std::clamp(value, min, max); };
}

} // namespace

PythonProperties::PythonProperties(nb::object ui) : ui(std::move(ui)) {}

void PythonProperties::rethrow_error() const {
    if (error) {
        std::rethrow_exception(error);
    }
}

template <typename... Args> nb::object PythonProperties::call(const char* method, Args&&... args) {
    if (error) {
        return nb::none();
    }
    try {
        return ui.attr(method)(std::forward<Args>(args)...);
    } catch (...) {
        error = std::current_exception();
        return nb::none();
    }
}

template <typename T, typename Adjust>
bool PythonProperties::assign(const nb::object& answer,
                              T* value,
                              const int components,
                              const Adjust& adjust) {
    if (answer.is_none()) {
        return false;
    }
    std::vector<T> read(components);
    try {
        for (int i = 0; i < components; i++) {
            read[i] = adjust(nb::cast<T>(components == 1 ? answer : answer[i]));
        }
    } catch (...) {
        error = std::current_exception();
        return false;
    }
    bool changed = false;
    for (int i = 0; i < components; i++) {
        changed |= read[i] != value[i];
        value[i] = read[i];
    }
    return changed;
}

bool PythonProperties::st_begin_child(const std::string& id,
                                      const std::string& label,
                                      const ChildFlags flags) {
    const nb::gil_scoped_acquire acquire;
    bool open = false;
    assign(call("begin_child", id, "label"_a = label.empty() ? id : label,
                "flags"_a = static_cast<ChildFlagBits>(flags)),
           &open, 1);
    return open;
}

void PythonProperties::st_end_child() {
    const nb::gil_scoped_acquire acquire;
    // Also after an error, the UI's children stay balanced
    try {
        ui.attr("end_child")();
    } catch (...) {
        if (!error) {
            error = std::current_exception();
        }
    }
}

void PythonProperties::st_separate(const std::string& label) {
    const nb::gil_scoped_acquire acquire;
    call("separate", label);
}

void PythonProperties::st_no_space() {
    const nb::gil_scoped_acquire acquire;
    call("no_space");
}

void PythonProperties::output_text(const std::string& text) {
    const nb::gil_scoped_acquire acquire;
    call("output_text", text);
}

void PythonProperties::output_plot_line(const std::string& label,
                                        const float* samples,
                                        const uint32_t count,
                                        const float scale_min,
                                        const float scale_max) {
    const nb::gil_scoped_acquire acquire;
    call("output_plot", label, std::vector<float>(samples, samples + count),
         "scale_min"_a = scale_min == FLT_MAX ? nb::none() : nb::cast(scale_min),
         "scale_max"_a = scale_max == FLT_MAX ? nb::none() : nb::cast(scale_max));
}

bool PythonProperties::config_float(const std::string& id,
                                    float* value,
                                    const std::string& desc,
                                    const int components,
                                    const float sensitivity,
                                    const std::optional<float>& min,
                                    const std::optional<float>& max) {
    const nb::gil_scoped_acquire acquire;
    return assign(call("config_float", id, to_python(value, components), "desc"_a = desc,
                       "sensitivity"_a = sensitivity, "minimum"_a = bound_to_python(min),
                       "maximum"_a = bound_to_python(max)),
                  value, components,
                  clamp_to(min.value_or(-std::numeric_limits<float>::infinity()),
                           max.value_or(std::numeric_limits<float>::infinity())));
}

template <typename T>
bool PythonProperties::config_integers(const std::string& id,
                                       T* value,
                                       const std::string& desc,
                                       const int components,
                                       const std::optional<T>& min,
                                       const std::optional<T>& max) {
    const nb::gil_scoped_acquire acquire;
    const T minimum = min.value_or(std::numeric_limits<T>::min());
    const T maximum = max.value_or(std::numeric_limits<T>::max());
    return assign(call("config_int", id, to_python(value, components), "desc"_a = desc,
                       "minimum"_a = minimum, "maximum"_a = maximum),
                  value, components, clamp_to(minimum, maximum));
}

bool PythonProperties::config_int(const std::string& id,
                                  int32_t* value,
                                  const std::string& desc,
                                  const int components,
                                  const std::optional<int32_t>& min,
                                  const std::optional<int32_t>& max) {
    return config_integers(id, value, desc, components, min, max);
}

bool PythonProperties::config_uint(const std::string& id,
                                   uint32_t* value,
                                   const std::string& desc,
                                   const int components,
                                   const std::optional<uint32_t>& min,
                                   const std::optional<uint32_t>& max) {
    return config_integers(id, value, desc, components, min, max);
}

bool PythonProperties::config_uint64(const std::string& id,
                                     uint64_t* value,
                                     const std::string& desc,
                                     const int components,
                                     const std::optional<uint64_t>& min,
                                     const std::optional<uint64_t>& max) {
    return config_integers(id, value, desc, components, min, max);
}

bool PythonProperties::config_angle(const std::string& id,
                                    float& angle,
                                    const std::string& desc,
                                    const float min,
                                    const float max) {
    float degrees = angle * DEGREES_PER_RADIAN;
    if (!config_float(id, degrees, desc, 0.1f, min, max)) {
        return false;
    }
    angle = degrees / DEGREES_PER_RADIAN;
    return true;
}

bool PythonProperties::config_split(const std::string& id,
                                    const std::vector<SplitPart>& parts,
                                    const std::string& desc) {
    bool reset = false;
    const std::vector<const SplitPart*> shown = shown_split_parts(parts, reset);
    if (shown.empty()) {
        return false;
    }
    std::vector<float> weights;
    const nb::gil_scoped_acquire acquire;
    nb::list labels;
    for (const SplitPart* part : shown) {
        weights.push_back(*part->weight);
        labels.append(part->label);
    }
    const int count = static_cast<int>(weights.size());
    const nb::object answer =
        call("config_split", id, labels, to_python(weights.data(), count), "desc"_a = desc);
    // A change made while other parts were shown
    if (!answer.is_none() && nb::len(answer) != weights.size()) {
        return reset;
    }
    if (!assign(answer, weights.data(), count,
                clamp_to(0.f, std::numeric_limits<float>::infinity()))) {
        return reset;
    }
    for (std::size_t i = 0; i < shown.size(); i++) {
        *shown[i]->weight = weights[i];
    }
    return true;
}

bool PythonProperties::config_color3(const std::string& id,
                                     float color[3],
                                     const std::string& desc) {
    const nb::gil_scoped_acquire acquire;
    return assign(call("config_color", id, to_python(color, 3), "desc"_a = desc), color, 3);
}

bool PythonProperties::config_color4(const std::string& id,
                                     float color[4],
                                     const std::string& desc) {
    const nb::gil_scoped_acquire acquire;
    return assign(call("config_color", id, to_python(color, 4), "desc"_a = desc), color, 4);
}

bool PythonProperties::config_bool(const std::string& id, bool& value, const std::string& desc) {
    const nb::gil_scoped_acquire acquire;
    return assign(call("config_bool", id, value, "desc"_a = desc), &value, 1);
}

bool PythonProperties::config_bool(const std::string& id, const std::string& desc) {
    const nb::gil_scoped_acquire acquire;
    bool pressed = false;
    assign(call("config_button", id, "desc"_a = desc), &pressed, 1);
    return pressed;
}

bool PythonProperties::config_options(const std::string& id,
                                      int& selected,
                                      const std::vector<std::string>& options,
                                      const OptionsStyle style,
                                      const std::string& desc) {
    const nb::gil_scoped_acquire acquire;
    if (options.empty()) {
        return false;
    }
    return assign(call("config_options", id, selected, "options"_a = options, "style"_a = style,
                       "desc"_a = desc),
                  &selected, 1, clamp_to(0, static_cast<int>(options.size()) - 1));
}

bool PythonProperties::config_text(const std::string& id,
                                   std::string& string,
                                   const bool needs_submit,
                                   const std::string& desc) {
    const nb::gil_scoped_acquire acquire;
    return assign(call("config_text", id, string, "multiline"_a = false,
                       "needs_submit"_a = needs_submit, "desc"_a = desc),
                  &string, 1);
}

bool PythonProperties::config_text_multiline(const std::string& id,
                                             std::string& string,
                                             const bool needs_submit,
                                             const std::string& desc) {
    const nb::gil_scoped_acquire acquire;
    return assign(call("config_text", id, string, "multiline"_a = true,
                       "needs_submit"_a = needs_submit, "desc"_a = desc),
                  &string, 1);
}

bool PythonProperties::config_path(const std::string& id,
                                   std::filesystem::path& path,
                                   const std::string& desc) {
    const nb::gil_scoped_acquire acquire;
    std::string text = path.string();
    if (!assign(call("config_path", id, text, "desc"_a = desc), &text, 1)) {
        return false;
    }
    path = text;
    return true;
}

bool PythonProperties::is_ui() {
    return true;
}

bool PythonProperties::serialize_json([[maybe_unused]] const std::string& id,
                                      [[maybe_unused]] nlohmann::json& json) {
    return false;
}

bool PythonProperties::serialize_string([[maybe_unused]] const std::string& id,
                                        [[maybe_unused]] std::string& s) {
    return false;
}

} // namespace merian::python
