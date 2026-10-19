#pragma once

// Python.h must precede the standard headers
#include <nanobind/nanobind.h>

#include "merian/utils/properties.hpp"

#include <exception>
#include <functional>

namespace merian::python {

// Records into a Python object with the methods of merian.Properties, taking the GIL per call.
// After the first error the pass records nothing more; rethrow_error() raises it.
class PythonProperties : public Properties {
  public:
    explicit PythonProperties(nanobind::object ui);

    void rethrow_error() const;

    bool st_begin_child(const std::string& id,
                        const std::string& label = "",
                        const ChildFlags flags = {}) override;
    void st_end_child() override;
    void st_separate(const std::string& label = "") override;
    void st_no_space() override;

    using Properties::output_text;
    void output_text(const std::string& text) override;
    void output_plot_line(const std::string& label,
                          const float* samples,
                          const uint32_t count,
                          const float scale_min = FLT_MAX,
                          const float scale_max = FLT_MAX) override;

    using Properties::config_bool;
    using Properties::config_float;
    using Properties::config_int;
    using Properties::config_uint;
    using Properties::config_uint64;
    bool config_float(const std::string& id,
                      float* value,
                      const std::string& desc = "",
                      const int components = 1,
                      const float sensitivity = 0.1f,
                      const std::optional<float>& min = std::nullopt,
                      const std::optional<float>& max = std::nullopt) override;
    bool config_int(const std::string& id,
                    int32_t* value,
                    const std::string& desc = "",
                    const int components = 1,
                    const std::optional<int32_t>& min = std::nullopt,
                    const std::optional<int32_t>& max = std::nullopt) override;
    bool config_uint(const std::string& id,
                     uint32_t* value,
                     const std::string& desc = "",
                     const int components = 1,
                     const std::optional<uint32_t>& min = std::nullopt,
                     const std::optional<uint32_t>& max = std::nullopt) override;
    bool config_uint64(const std::string& id,
                       uint64_t* value,
                       const std::string& desc = "",
                       const int components = 1,
                       const std::optional<uint64_t>& min = std::nullopt,
                       const std::optional<uint64_t>& max = std::nullopt) override;
    bool config_angle(const std::string& id,
                      float& angle,
                      const std::string& desc = "",
                      const float min = -360,
                      const float max = 360) override;
    bool config_split(const std::string& id,
                      const std::vector<SplitPart>& parts,
                      const std::string& desc = "") override;
    bool
    config_color3(const std::string& id, float color[3], const std::string& desc = "") override;
    bool
    config_color4(const std::string& id, float color[4], const std::string& desc = "") override;
    bool config_bool(const std::string& id, bool& value, const std::string& desc = "") override;
    bool config_bool(const std::string& id, const std::string& desc = "") override;
    bool config_options(const std::string& id,
                        int& selected,
                        const std::vector<std::string>& options,
                        const OptionsStyle style = OptionsStyle::DONT_CARE,
                        const std::string& desc = "") override;
    bool config_text(const std::string& id,
                     std::string& string,
                     const bool needs_submit = false,
                     const std::string& desc = "") override;
    bool config_text_multiline(const std::string& id,
                               std::string& string,
                               const bool needs_submit = false,
                               const std::string& desc = "") override;
    using Properties::config_path;
    bool config_path(const std::string& id,
                     std::filesystem::path& path,
                     const std::string& desc = "") override;

    bool is_ui() override;
    bool serialize_json(const std::string& id, nlohmann::json& json) override;
    bool serialize_string(const std::string& id, std::string& s) override;

  private:
    template <typename... Args> nanobind::object call(const char* method, Args&&... args);

    template <typename T, typename Adjust = std::identity>
    bool assign(const nanobind::object& answer,
                T* value,
                const int components,
                const Adjust& adjust = {});

    template <typename T>
    bool config_integers(const std::string& id,
                         T* value,
                         const std::string& desc,
                         const int components,
                         const std::optional<T>& min,
                         const std::optional<T>& max);

    nanobind::object ui;
    std::exception_ptr error;
};

} // namespace merian::python
