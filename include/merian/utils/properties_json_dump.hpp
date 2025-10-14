#pragma once

#include "nlohmann/json.hpp"
#include "properties.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace merian {

class JSONDumpProperties : public Properties {
  public:
    // If filename is not nullopt the configuration is dumped in the destructor.
    JSONDumpProperties(const std::optional<std::filesystem::path>& filename = std::nullopt);

    virtual ~JSONDumpProperties() override;

    virtual bool st_begin_child(const std::string& id,
                                const std::string& label = "",
                                const ChildFlags flags = {}) override;
    virtual void st_end_child() override;

    virtual void st_separate(const std::string& label = "") override;
    virtual void st_no_space() override;

    virtual void output_text(const std::string& text) override;
    virtual void output_plot_line(const std::string& label,
                                  const float* samples,
                                  const uint32_t count,
                                  const float scale_min,
                                  const float scale_max) override;

    virtual bool config_float(const std::string& id,
                              float* value,
                              const std::string& desc = "",
                              const int components = 1) override;
    virtual bool config_int(const std::string& id,
                            int32_t* value,
                            const std::string& desc = "",
                            const int components = 1) override;
    virtual bool config_uint(const std::string& id,
                             uint32_t* value,
                             const std::string& desc = "",
                             const int components = 1) override;
    virtual bool config_uint64(const std::string& id,
                               uint64_t* value,
                               const std::string& desc = "",
                               const int components = 1) override;

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

    nlohmann::json get() const {
        assert(o.size() == 1 && "Missing st_end_child?");
        return current();
    }

    std::string string() const {
        assert(o.size() == 1 && "Missing st_end_child?");
        return current().dump();
    }

  private:
    nlohmann::json& current() {
        return o.back().second;
    }

    const nlohmann::json& current() const {
        return o.back().second;
    }

  private:
    const std::optional<std::filesystem::path> filename;
    std::vector<std::pair<std::string, nlohmann::json>> o;
};

} // namespace merian
