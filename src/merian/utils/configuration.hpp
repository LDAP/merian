#pragma once

#include "glm/glm.hpp"

#include <string>
#include <vector>

namespace merian {

// "Record" configurtation options and information to display.
// Some implementations will not allowed that parameters called `id` have the same name in the same
// child.
class Configuration {
  public:
    enum class OptionsStyle {
        DONT_CARE,
        RADIO_BUTTON,
        COMBO,
        LIST_BOX,
    };

    virtual ~Configuration() {};

    // Structure

    // Returns true if the child should be examined. Call st_end_child at the end of the
    // section if true was returned. 
    [[nodiscard]] virtual bool st_begin_child(const std::string& id,
                                              const std::string& label = "") = 0;
    // Must only be called if begin_child was true
    virtual void st_end_child() = 0;

    // Starts a new configuration section, that ends at the next st_end_child.
    // This has no meaning when identifying the configuration option
    // but can structure config when displayed.
    // Retuns true if the section should be examined.
    virtual bool st_new_section(const std::string& label) = 0;

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
    virtual void output_plot_line(const std::string& label,
                                  const std::vector<float>& samples,
                                  const float scale_min = FLT_MAX,
                                  const float scale_max = FLT_MAX) = 0;

    // Config

    virtual void config_color(const std::string& id, glm::vec3& color) = 0;
    virtual void config_color(const std::string& id, glm::vec4& color) = 0;
    virtual void config_vec(const std::string& id, glm::vec3& value) = 0;
    virtual void config_vec(const std::string& id, glm::vec4& value) = 0;
    virtual void config_angle(const std::string& id, float& angle) = 0;
    virtual void config_percent(const std::string& id, float& value) = 0;

    virtual void config_float(const std::string& id, float& value) = 0;
    virtual void config_float(const std::string& id,
                              float& value,
                              const float& min,
                              const float& max) = 0;
    virtual void config_int(const std::string& id, int& value) = 0;
    virtual void config_int(const std::string& id,
                            int& value,
                            const int& min = std::numeric_limits<int>::min(),
                            const int& max = std::numeric_limits<int>::max()) = 0;
    virtual void config_float3(const std::string& id, float value[3]) = 0;
    // Holds the supplied `value` if not changed by the configuration.
    // Converts to a checkbox in a GUI context.
    virtual void config_bool(const std::string& id, bool& value) = 0;
    // If set by the configuration returns true only once `one-shot`.
    // Converts to a button in a GUI context.
    [[nodiscard]] virtual bool config_bool(const std::string& id) = 0;
    virtual void config_options(const std::string& id,
                                int& selected,
                                const std::vector<std::string>& options,
                                const OptionsStyle style = OptionsStyle::DONT_CARE) = 0;
    // If `needs_submit` is true then the user can enter the text and then explicitly submit in a
    // GUI context (e.g using a button or by pressing enter). If `needs_submit` is false, then true
    // is returned at every change.
    [[nodiscard]] virtual bool config_text(const std::string& id,
                                           const uint32_t max_len,
                                           char* string,
                                           const bool needs_submit = false) = 0;
};

} // namespace merian
