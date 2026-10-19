#pragma once

#include "merian/utils/properties.hpp"
#include "nlohmann/json.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace merian {

class JSONLoadProperties : public Properties {
  public:
    JSONLoadProperties(const std::filesystem::path& filename);

    JSONLoadProperties(const std::string& json_string);

    JSONLoadProperties(const nlohmann::json& json);

    JSONLoadProperties(const nlohmann::json&& json);

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

    virtual bool config_float(const std::string& id,
                              float* value,
                              const std::string& desc = "",
                              const int components = 1,
                              const float sensitivity = 0.1f,
                              const std::optional<float>& min = std::nullopt,
                              const std::optional<float>& max = std::nullopt) override;
    virtual bool config_int(const std::string& id,
                            int32_t* value,
                            const std::string& desc = "",
                            const int components = 1,
                            const std::optional<int32_t>& min = std::nullopt,
                            const std::optional<int32_t>& max = std::nullopt) override;
    virtual bool config_uint(const std::string& id,
                             uint32_t* value,
                             const std::string& desc = "",
                             const int components = 1,
                             const std::optional<uint32_t>& min = std::nullopt,
                             const std::optional<uint32_t>& max = std::nullopt) override;
    virtual bool config_uint64(const std::string& id,
                               uint64_t* value,
                               const std::string& desc = "",
                               const int components = 1,
                               const std::optional<uint64_t>& min = std::nullopt,
                               const std::optional<uint64_t>& max = std::nullopt) override;

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

    // JSON pointers of the keys no property read.
    std::vector<std::string> unread_keys() const;

    // Options whose value is none of the choices.
    const std::vector<std::string>& invalid_values() const;

    // The invalid values and, if any, "unknown <unread keys>".
    std::vector<std::string> problems() const;

    // Loads again with the keys the first load left unread (they may have been revealed), unless
    // it returned false. Returns the problems() of both and the error that aborted a load.
    static std::vector<std::string> load_revealed(const nlohmann::json& json,
                                                  const std::function<bool(Properties&)>& load);

  private:
    bool read(const std::string& id);

    void collect_unread(const nlohmann::json& object,
                        const nlohmann::json::json_pointer& pointer,
                        std::vector<std::string>& unread) const;

    std::string object_name;
    std::vector<nlohmann::json> o;

    std::vector<bool> transparent;

    nlohmann::json::json_pointer current;
    std::unordered_set<std::string> read_keys;
    std::unordered_set<std::string> entered_children;
    std::vector<std::string> invalid;
};

} // namespace merian
