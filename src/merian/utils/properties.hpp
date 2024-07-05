#pragma once

#include "ext/json.hpp"
#include "glm/glm.hpp"
#include "merian/utils/enums.hpp"

#include <fmt/format.h>
#include <string>
#include <vector>

namespace merian {

// "Record" configurtation options and information to display.
// Some implementations will not allowed that parameters called `id` have the same name in the same
// child.
// Different recorders can for example display the configuration in a GUI,
// dump it to a file or load a dump from a file.
class Properties {
  public:
    enum class OptionsStyle {
        DONT_CARE,
        RADIO_BUTTON,
        COMBO,
        LIST_BOX,
    };

    enum ChildFlagBits : uint32_t {
        DEFAULT_OPEN = 0b1,
        FRAMED = 0b10,
    };

    using ChildFlags = uint32_t;

    virtual ~Properties(){};

    // Structure

    // Returns true if the child should be examined. Call st_end_child at the end of the
    // section if true was returned.
    [[nodiscard]] virtual bool st_begin_child(const std::string& id,
                                              const std::string& label = "",
                                              const ChildFlags flags = {}) = 0;
    // Must only be called if begin_child was true
    virtual void st_end_child() = 0;

    // Separates config options.
    // This has no meaning when identifying the configuration option
    // but can structure config when displayed.
    virtual void st_separate(const std::string& label = "") = 0;
    // Attemps to keep output and/or config together, e.g. by displaying on the same line
    // This has no meaning when identifying the configuration option
    // but can structure config when displayed.
    virtual void st_no_space() = 0;

    // Output

    virtual void output_text(const std::string& text) = 0;
    template <typename... T> void output_text(fmt::format_string<T...> fmt, T&&... args) {
        output_text(fmt::format(fmt, args...));
    }
    virtual void output_plot_line(const std::string& label,
                                  const float* samples,
                                  const uint32_t count,
                                  const float scale_min = FLT_MAX,
                                  const float scale_max = FLT_MAX) = 0;

    // Config

    // Returns true if the value changed.
    virtual bool
    config_color(const std::string& id, glm::vec3& color, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool
    config_color(const std::string& id, glm::vec4& color, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, glm::vec3& value, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, glm::vec4& value, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, glm::uvec3& value, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, glm::uvec4& value, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool config_angle(const std::string& id,
                              float& angle,
                              const std::string& desc = "",
                              const float min = -360,
                              const float max = 360) = 0;
    // Returns true if the value changed.
    virtual bool
    config_percent(const std::string& id, float& value, const std::string& desc = "") = 0;

    // Returns true if the value changed.
    virtual bool config_float(const std::string& id,
                              float& value,
                              const std::string& desc = "",
                              const float sensitivity = 1.0f) = 0;
    // Returns true if the value changed.
    virtual bool config_float(const std::string& id,
                              float& value,
                              const float& min,
                              const float& max,
                              const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool config_int(const std::string& id, int& value, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool config_int(const std::string& id,
                            int& value,
                            const int& min = std::numeric_limits<int>::min(),
                            const int& max = std::numeric_limits<int>::max(),
                            const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool
    config_uint(const std::string& id, uint32_t& value, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool config_uint(const std::string& id,
                             uint32_t& value,
                             const uint32_t& min = std::numeric_limits<uint32_t>::min(),
                             const uint32_t& max = std::numeric_limits<uint32_t>::max(),
                             const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool
    config_float3(const std::string& id, float value[3], const std::string& desc = "") = 0;
    // Holds the supplied `value` if not changed by the configuration.
    // Converts to a checkbox in a GUI context.
    // Returns true if the value changed.
    virtual bool config_bool(const std::string& id, int& value, const std::string& desc = "") {
        bool bool_value = value;
        const bool changed = config_bool(id, bool_value, desc);
        value = bool_value;
        return changed;
    }
    // Holds the supplied `value` if not changed by the configuration.
    // Converts to a checkbox in a GUI context.
    // Returns true if the value changed.
    virtual bool config_bool(const std::string& id, bool& value, const std::string& desc = "") = 0;
    // Returns true if the value changed.
    virtual bool config_options(const std::string& id,
                                int& selected,
                                const std::vector<std::string>& options,
                                const OptionsStyle style = OptionsStyle::DONT_CARE,
                                const std::string& desc = "") = 0;

    // Needs specializations for the enum_size(), enum_values() and enum_to_string() methods.
    template <typename ENUM_TYPE>
    bool config_enum(const std::string& id,
                     ENUM_TYPE& value,
                     const OptionsStyle style = OptionsStyle::DONT_CARE,
                     const std::string& desc = "") {
        std::vector<std::string> options;
        int selected = 0;
        int i = 0;
        for (const ENUM_TYPE* p = enum_values<ENUM_TYPE>();
             p < enum_values<ENUM_TYPE>() + enum_size<ENUM_TYPE>(); p++) {
            options.emplace_back(enum_to_string(*p));
            if (*p == value) {
                selected = i;
            }
            i++;
        }

        const bool value_changed = config_options(id, selected, options, style, desc);
        value = enum_values<ENUM_TYPE>()[selected];

        return value_changed;
    }

    // If set by the configuration returns true only once `one-shot`.
    // Converts to a button in a GUI context.
    // Note that this behavior is different to most config_ methods.
    [[nodiscard]] virtual bool config_bool(const std::string& id, const std::string& desc = "") = 0;

    // If `needs_submit` is true then the user can enter the text and then explicitly submit in a
    // GUI context (e.g using a button or by pressing enter). If `needs_submit` is false, then true
    // is returned at every change.
    [[nodiscard]] virtual bool config_text(const std::string& id,
                                           const uint32_t max_len,
                                           char* string,
                                           const bool needs_submit = false,
                                           const std::string& desc = "") = 0;
    [[nodiscard]] virtual bool config_text_multiline(const std::string& id,
                                                     const uint32_t max_len,
                                                     char* string,
                                                     const bool needs_submit = false,
                                                     const std::string& desc = "") = 0;

    // Serialization

    // Serializaion allows to store and load data. These possibly not shown in the UI.

    // Returns true if you should provide valid data to the serialize_* methods. If this is false no
    // data is stored and you can skip computing the data. serialize_* methods may still return
    // true, meaning that you should load new data.
    [[nodiscard]]
    virtual bool serialize() = 0;

    // Returns true, if new data was loaded.
    virtual bool serialize_json(const std::string& id, nlohmann::json& json) = 0;

    // Returns true, if new data was loaded.
    virtual bool serialize_string(const std::string& id, std::string& s) = 0;
};

} // namespace merian
