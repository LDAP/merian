#pragma once

#include "ext/json.hpp"
#include "merian/utils/properties.hpp"

#include <string>
#include <vector>

namespace merian {

class JSONLoadProperties : public Properties {
  public:
    JSONLoadProperties(const std::filesystem::path& filename);

    JSONLoadProperties(const std::string& json_string);

    virtual ~JSONLoadProperties() override;

    virtual bool st_begin_child(const std::string& id,
                                const std::string& label = "",
                                const ChildFlags flags = {}) override;
    virtual void st_end_child() override;
    virtual std::vector<std::string> st_list_children() override;

    virtual void st_separate(const std::string& label = "") override;
    virtual void st_no_space() override;

    virtual void output_text(const std::string& text) override;
    virtual void output_plot_line(const std::string& label,
                                  const float* samples,
                                  const uint32_t count,
                                  const float scale_min,
                                  const float scale_max) override;

    virtual bool
    config_color(const std::string& id, glm::vec3& color, const std::string& desc = "") override;
    virtual bool
    config_color(const std::string& id, glm::vec4& color, const std::string& desc = "") override;
    virtual bool
    config_vec(const std::string& id, glm::vec3& value, const std::string& desc = "") override;
    virtual bool
    config_vec(const std::string& id, glm::vec4& value, const std::string& desc = "") override;
    virtual bool
    config_vec(const std::string& id, glm::uvec3& value, const std::string& desc = "") override;
    virtual bool
    config_vec(const std::string& id, glm::uvec4& value, const std::string& desc = "") override;
    virtual bool config_angle(const std::string& id,
                              float& angle,
                              const std::string& desc = "",
                              const float min = -360,
                              const float max = 360) override;
    virtual bool
    config_percent(const std::string& id, float& value, const std::string& desc = "") override;
    virtual bool config_float(const std::string& id,
                              float& value,
                              const std::string& desc = "",
                              const float sensitivity = 1.0f) override;
    virtual bool config_float(const std::string& id,
                              float& value,
                              const float& min = FLT_MIN,
                              const float& max = FLT_MAX,
                              const std::string& desc = "") override;
    virtual bool
    config_int(const std::string& id, int& value, const std::string& desc = "") override;
    virtual bool config_int(const std::string& id,
                            int& value,
                            const int& min = std::numeric_limits<int>::min(),
                            const int& max = std::numeric_limits<int>::max(),
                            const std::string& desc = "") override;
    virtual bool
    config_uint(const std::string& id, uint32_t& value, const std::string& desc = "") override;
    virtual bool config_uint(const std::string& id,
                             uint32_t& value,
                             const uint32_t& min = std::numeric_limits<uint32_t>::min(),
                             const uint32_t& max = std::numeric_limits<uint32_t>::max(),
                             const std::string& desc = "") override;
    virtual bool
    config_float3(const std::string& id, float value[3], const std::string& desc = "") override;
    virtual bool
    config_bool(const std::string& id, bool& value, const std::string& desc = "") override;
    virtual bool config_bool(const std::string& id, const std::string& desc = "") override;
    virtual bool config_options(const std::string& id,
                                int& selected,
                                const std::vector<std::string>& options,
                                const OptionsStyle style = OptionsStyle::DONT_CARE,
                                const std::string& desc = "") override;
    virtual bool config_text(const std::string& id,
                             std::string& string,
                             const bool needs_submit = false,
                             const std::string& desc = "") override;
    virtual bool config_text_multiline(const std::string& id,
                                       std::string& string,
                                       const bool needs_submit = false,
                                       const std::string& desc = "") override;

    virtual bool is_ui() override;
    virtual bool serialize_json(const std::string& id, nlohmann::json& json) override;
    virtual bool serialize_string(const std::string& id, std::string& s) override;

  private:
    std::string object_name;
    std::vector<nlohmann::json> o;
};

} // namespace merian
