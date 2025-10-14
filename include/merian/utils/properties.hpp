#pragma once

#include "merian/utils/enums.hpp"
#include "merian/utils/vector_matrix.hpp"

#include "nlohmann/json.hpp"

#include <cfloat>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace merian {

// "Record" configurtation options and information to display.
// Some implementations will not allow that parameters called `id` have the same name in the same
// child. Different recorders can for example display the configuration in a GUI,
// dump it to a file or load a dump from a file.
//
// Empty IDs are allowed when "is_ui" returnes true, otherwise it depends on the implementation or
// may lead to undefined behavior. Empty IDs are never allowed at st_begin_child().
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

    virtual ~Properties() {};

    // Structure

    // Returns true if the child should be examined. Call st_end_child at the end of the
    // section if true was returned.
    [[nodiscard]] virtual bool st_begin_child(const std::string& id,
                                              const std::string& label = "",
                                              const ChildFlags flags = {}) = 0;
    // Must only be called if begin_child was true
    virtual void st_end_child() = 0;

    // List known children, if supported. This is useful as "lookahead" when first loading.
    virtual std::vector<std::string> st_list_children() {
        return {};
    }

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
        output_text(vformat(fmt, fmt::make_format_args(args...)));
    }
    virtual void output_plot_line(const std::string& label,
                                  const float* samples,
                                  const uint32_t count,
                                  const float scale_min = FLT_MAX,
                                  const float scale_max = FLT_MAX) = 0;

    // Config

    // Returns true if the value changed.
    virtual bool config_float(const std::string& id,
                              float* value,
                              const std::string& desc = "",
                              const int components = 1) = 0;
    // Returns true if the value changed.
    virtual bool config_int(const std::string& id,
                            int32_t* value,
                            const std::string& desc = "",
                            const int components = 1) = 0;
    // Returns true if the value changed.
    virtual bool config_uint(const std::string& id,
                             uint32_t* value,
                             const std::string& desc = "",
                             const int components = 1) = 0;
    // Returns true if the value changed.
    virtual bool config_uint64(const std::string& id,
                               uint64_t* value,
                               const std::string& desc = "",
                               const int components = 1) = 0;

    // Returns true if the value changed.
    virtual bool config_float(const std::string& id,
                              float& value,
                              const std::string& desc = "",
                              const float /*sensitivity*/ = 1.0f) {
        return config_float(id, &value, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool config_float(const std::string& id,
                              float& value,
                              const float& /*min*/,
                              const float& /*max*/,
                              const std::string& desc = "") {
        return config_float(id, &value, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool
    config_int(const std::string& id, int32_t& value, const std::string& desc = "") final {
        return config_int(id, &value, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool config_int(const std::string& id,
                            int& value,
                            const int& /*min*/,
                            const int& /*max*/,
                            const std::string& desc = "") {
        return config_int(id, &value, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool
    config_uint(const std::string& id, uint32_t& value, const std::string& desc = "") final {
        return config_uint(id, &value, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool config_uint(const std::string& id,
                             uint32_t& value,
                             const uint32_t& /*min*/,
                             const uint32_t& /*max*/,
                             const std::string& desc = "") {
        return config_uint(id, &value, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool
    config_uint64(const std::string& id, uint64_t& value, const std::string& desc = "") final {
        return config_uint64(id, &value, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool config_uint64(const std::string& id,
                               uint64_t& value,
                               const uint64_t& /*min*/,
                               const uint64_t& /*max*/,
                               const std::string& desc = "") {
        return config_uint64(id, &value, desc, 1);
    }

    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, float1& value, const std::string& desc = "") final {
        return config_float(id, &value.x, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, float2& value, const std::string& desc = "") final {
        return config_float(id, &value.x, desc, 2);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, float3& value, const std::string& desc = "") final {
        return config_float(id, &value.x, desc, 3);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, float4& value, const std::string& desc = "") final {
        return config_float(id, &value.x, desc, 4);
    }

    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, int1& value, const std::string& desc = "") final {
        return config_int(id, &value.x, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, int2& value, const std::string& desc = "") final {
        return config_int(id, &value.x, desc, 2);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, int3& value, const std::string& desc = "") final {
        return config_int(id, &value.x, desc, 3);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, int4& value, const std::string& desc = "") final {
        return config_int(id, &value.x, desc, 4);
    }

    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, uint1& value, const std::string& desc = "") final {
        return config_uint(id, &value.x, desc, 1);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, uint2& value, const std::string& desc = "") final {
        return config_uint(id, &value.x, desc, 2);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, uint3& value, const std::string& desc = "") final {
        return config_uint(id, &value.x, desc, 3);
    }
    // Returns true if the value changed.
    virtual bool
    config_vec(const std::string& id, uint4& value, const std::string& desc = "") final {
        return config_uint(id, &value.x, desc, 4);
    }

    // Returns true if the value changed.
    virtual bool
    config_color3(const std::string& id, float color[3], const std::string& desc = "") {
        return config_float(id, color, desc, 3);
    }
    // Returns true if the value changed.
    virtual bool
    config_color4(const std::string& id, float color[4], const std::string& desc = "") {
        return config_float(id, color, desc, 4);
    }

    // Returns true if the value changed.
    virtual bool
    config_color(const std::string& id, float3& color, const std::string& desc = "") final {
        return config_color3(id, &color.x, desc);
    }
    // Returns true if the value changed.
    virtual bool
    config_color(const std::string& id, float4& color, const std::string& desc = "") final {
        return config_color4(id, &color.x, desc);
    }

    // Returns true if the value changed.
    virtual bool config_angle(const std::string& id,
                              float& angle,
                              const std::string& desc = "",
                              const float min = -360,
                              const float max = 360) {
        return config_float(id, angle, min, max, desc);
    }

    // Returns true if the value changed.
    virtual bool config_percent(const std::string& id, float& value, const std::string& desc = "") {
        return config_float(id, value, 0.f, 1.f, desc);
    }

    // Holds the supplied `value` if not changed by the configuration.
    // Converts to a checkbox in a GUI context.
    // Returns true if the value changed.
    virtual bool config_bool(const std::string& id, bool& value, const std::string& desc = "") = 0;
    // Holds the supplied `value` if not changed by the configuration.
    // Converts to a checkbox in a GUI context.
    // Returns true if the value changed.
    virtual bool config_bool(const std::string& id, int32_t& value, const std::string& desc = "") {
        bool bool_value = value != 0;
        if (config_bool(id, bool_value, desc)) {
            value = static_cast<int32_t>(bool_value);
            return true;
        }
        return false;
    }
    // Holds the supplied `value` if not changed by the configuration.
    // Converts to a checkbox in a GUI context.
    // Returns true if the value changed.
    virtual bool config_bool(const std::string& id, uint32_t& value, const std::string& desc = "") {
        bool bool_value = value != 0u;
        if (config_bool(id, bool_value, desc)) {
            value = static_cast<uint32_t>(bool_value);
            return true;
        }
        return false;
    }
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
                                           std::string& string,
                                           const bool needs_submit = false,
                                           const std::string& desc = "") = 0;
    [[nodiscard]] virtual bool config_text_multiline(const std::string& id,
                                                     std::string& string,
                                                     const bool needs_submit = false,
                                                     const std::string& desc = "") = 0;

    // Serialization

    // Serializaion allows to store and load data. These possibly not shown in the UI.

    // Returns true if the Properties object is a UI interface. You can use that to selectively hide
    // elements that should only be shown on UI or serialized.
    [[nodiscard]]
    virtual bool is_ui() = 0;

    // Returns true, if new data was loaded.
    virtual bool serialize_json(const std::string& id, nlohmann::json& json) = 0;

    // Returns true, if new data was loaded.
    virtual bool serialize_string(const std::string& id, std::string& s) = 0;
};

} // namespace merian
