#include "properties_imgui.hpp"

#include "imgui.h"
#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace merian {

void tooltip(const std::string& tooltip) {
    if (!tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("%s", tooltip.c_str());
    }
}

ImGuiProperties::~ImGuiProperties() {}

bool ImGuiProperties::st_begin_child(const std::string& id, const std::string& label) {
    return ImGui::TreeNode(id.c_str(), "%s", label.c_str());
}
void ImGuiProperties::st_end_child() {
    ImGui::TreePop();
}
bool ImGuiProperties::st_new_section(const std::string& label) {
    return ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
}
void ImGuiProperties::st_separate(const std::string& label) {
    if (label.empty())
        ImGui::Separator();
    else
        ImGui::SeparatorText(label.c_str());
}
void ImGuiProperties::st_no_space() {
    ImGui::SameLine();
}

void ImGuiProperties::output_text(const std::string& text) {
    ImGui::TextWrapped("%s", text.c_str());
}
void ImGuiProperties::output_plot_line(const std::string& label,
                                       const float* samples,
                                       const uint32_t count,
                                       const float scale_min,
                                       const float scale_max) {
    ImGui::PlotLines(label.c_str(), samples, count, 0, NULL, scale_min, scale_max,
                     {0, ImGui::GetFontSize() * 5});
}

bool ImGuiProperties::config_color(const std::string& id,
                                   glm::vec3& color,
                                   const std::string& desc) {
    const bool value_changed = ImGui::ColorEdit3(id.c_str(), &color.x);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_color(const std::string& id,
                                   glm::vec4& color,
                                   const std::string& desc) {
    const bool value_changed = ImGui::ColorEdit4(id.c_str(), &color.x);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_vec(const std::string& id, glm::vec3& value, const std::string& desc) {
    const bool value_changed = ImGui::InputFloat3(id.c_str(), &value.x);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_vec(const std::string& id, glm::vec4& value, const std::string& desc) {
    const bool value_changed = ImGui::InputFloat4(id.c_str(), &value.x);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_vec(const std::string& id,
                                 glm::uvec3& value,
                                 const std::string& desc) {
    const bool value_changed = ImGui::InputScalarN(id.c_str(), ImGuiDataType_U32, &value.x, 3);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_vec(const std::string& id,
                                 glm::uvec4& value,
                                 const std::string& desc) {
    const bool value_changed = ImGui::InputScalarN(id.c_str(), ImGuiDataType_U32, &value.x, 4);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_angle(const std::string& id,
                                   float& angle,
                                   const std::string& desc,
                                   const float min,
                                   const float max) {
    const bool value_changed = ImGui::SliderAngle(id.c_str(), &angle, min, max);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_percent(const std::string& id, float& value, const std::string& desc) {
    const bool value_changed = ImGui::SliderFloat(id.c_str(), &value, 0, 1, "%.06f");
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_float(const std::string& id,
                                   float& value,
                                   const std::string& desc,
                                   const float sensitivity) {
    const bool value_changed = ImGui::DragFloat(
        id.c_str(), &value, sensitivity, 0.0f, 0.0f,
        fmt::format("%.{}f", std::max(0, (int)std::ceil(-std::log10(sensitivity)))).c_str());
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_float(const std::string& id,
                                   float& value,
                                   const float& min,
                                   const float& max,
                                   const std::string& desc) {
    const bool value_changed = ImGui::SliderFloat(id.c_str(), &value, min, max);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_int(const std::string& id, int& value, const std::string& desc) {
    const bool value_changed = ImGui::DragInt(id.c_str(), &value);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_int(
    const std::string& id, int& value, const int& min, const int& max, const std::string& desc) {
    const bool value_changed = ImGui::SliderInt(id.c_str(), &value, min, max);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_uint(const std::string& id, uint32_t& value, const std::string& desc) {
    const bool value_changed = ImGui::DragScalar(id.c_str(), ImGuiDataType_U32, &value);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_uint(const std::string& id,
                                  uint32_t& value,
                                  const uint32_t& min,
                                  const uint32_t& max,
                                  const std::string& desc) {
    const bool value_changed =
        ImGui::SliderScalar(id.c_str(), ImGuiDataType_U32, &value, &min, &max);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_float3(const std::string& id,
                                    float value[3],
                                    const std::string& desc) {
    const bool value_changed = ImGui::InputFloat3(id.c_str(), value);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_bool(const std::string& id, bool& value, const std::string& desc) {
    const bool old_value = value;
    ImGui::Checkbox(id.c_str(), &value);
    tooltip(desc);
    return old_value != value;
}
bool ImGuiProperties::config_bool(const std::string& id, const std::string& desc) {
    bool pressed = ImGui::Button(id.c_str());
    tooltip(desc);
    return pressed;
}
bool ImGuiProperties::config_options(const std::string& id,
                                     int& selected,
                                     const std::vector<std::string>& options,
                                     const OptionsStyle style,
                                     const std::string& desc) {
    const int old_selected = selected;

    switch (style) {
    case OptionsStyle::RADIO_BUTTON:
        for (uint32_t i = 0; i < options.size(); i++) {
            ImGui::RadioButton(options[i].c_str(), &selected, i);
            tooltip(desc);
        }
        break;
    case OptionsStyle::COMBO:
        ImGui::Combo(
            id.c_str(), &selected,
            [](void* data, int n, const char** out_str) {
                const std::vector<std::string>* options =
                    reinterpret_cast<std::vector<std::string>*>(data);
                *out_str = (*options)[n].c_str();
                return true;
            },
            (void*)(&options), options.size());
        tooltip(desc);
        break;
    case OptionsStyle::DONT_CARE:
    case OptionsStyle::LIST_BOX: {
        std::vector<const char*> options_c_str;
        std::transform(options.begin(), options.end(), std::back_inserter(options_c_str),
                       [](auto& str) { return str.c_str(); });
        ImGui::ListBox(id.c_str(), &selected, options_c_str.data(), options_c_str.size());
        tooltip(desc);
        break;
    }
    default:
        throw std::runtime_error{"OptionsStyle not supported"};
    }

    return old_selected != selected;
}
bool ImGuiProperties::config_text(const std::string& id,
                                  const uint32_t max_len,
                                  char* string,
                                  const bool needs_submit,
                                  const std::string& desc) {

    bool submit_change = ImGui::InputText(id.c_str(), string, max_len,
                                          needs_submit ? ImGuiInputTextFlags_EnterReturnsTrue : 0);
    tooltip(desc);
    return submit_change;
}

bool ImGuiProperties::config_text_multiline(const std::string& id,
                                            const uint32_t max_len,
                                            char* string,
                                            const bool needs_submit,
                                            const std::string& desc) {
    bool submit_change =
        ImGui::InputTextMultiline(id.c_str(), string, max_len, ImVec2(0, 0),
                                  needs_submit ? ImGuiInputTextFlags_EnterReturnsTrue : 0);
    tooltip(desc);
    return submit_change;
}

bool ImGuiProperties::serialize() {
    return false;
}
bool ImGuiProperties::serialize_json(const std::string&, nlohmann::json&) {
    return false;
}
bool ImGuiProperties::serialize_string(const std::string&, std::string&) {
    return false;
}

} // namespace merian
