#include "configuration_imgui.hpp"
#include "imgui.h"
#include <algorithm>
#include <fmt/format.h>
#include <stdexcept>

namespace merian {

void tooltip(const std::string& tooltip) {
    if (!tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("%s", tooltip.c_str());
    }
}

ImGuiConfiguration::~ImGuiConfiguration() {}

bool ImGuiConfiguration::st_begin_child(const std::string& id, const std::string& label) {
    return ImGui::TreeNode(id.c_str(), "%s", label.c_str());
}
void ImGuiConfiguration::st_end_child() {
    ImGui::TreePop();
}
bool ImGuiConfiguration::st_new_section(const std::string& label) {
    return ImGui::CollapsingHeader(label.c_str());
}
void ImGuiConfiguration::st_separate(const std::string& label) {
    if (label.empty())
        ImGui::Separator();
    else
        ImGui::SeparatorText(label.c_str());
}
void ImGuiConfiguration::st_no_space() {
    ImGui::SameLine();
}

void ImGuiConfiguration::output_text(const std::string& text) {
    ImGui::TextWrapped("%s", text.c_str());
}
void ImGuiConfiguration::output_plot_line(const std::string& label,
                                          const std::vector<float>& samples,
                                          const float scale_min,
                                          const float scale_max) {
    ImGui::PlotLines(label.c_str(), samples.data(), samples.size(), 0, NULL, scale_min, scale_max);
}

void ImGuiConfiguration::config_color(const std::string& id,
                                      glm::vec3& color,
                                      const std::string& desc) {
    ImGui::ColorEdit3(id.c_str(), &color.x);
    tooltip(desc);
}
void ImGuiConfiguration::config_color(const std::string& id,
                                      glm::vec4& color,
                                      const std::string& desc) {
    ImGui::ColorEdit4(id.c_str(), &color.x);
    tooltip(desc);
}
void ImGuiConfiguration::config_vec(const std::string& id,
                                    glm::vec3& value,
                                    const std::string& desc) {
    ImGui::InputFloat3(id.c_str(), &value.x);
    tooltip(desc);
}
void ImGuiConfiguration::config_vec(const std::string& id,
                                    glm::vec4& value,
                                    const std::string& desc) {
    ImGui::InputFloat4(id.c_str(), &value.x);
    tooltip(desc);
}
void ImGuiConfiguration::config_angle(const std::string& id,
                                      float& angle,
                                      const std::string& desc,
                                      const float min,
                                      const float max) {
    ImGui::SliderAngle(id.c_str(), &angle, min, max);
    tooltip(desc);
}
void ImGuiConfiguration::config_percent(const std::string& id,
                                        float& value,
                                        const std::string& desc) {
    ImGui::SliderFloat(id.c_str(), &value, 0, 1, "%.06f");
    tooltip(desc);
}
void ImGuiConfiguration::config_float(const std::string& id,
                                      float& value,
                                      const std::string& desc,
                                      const float sensitivity) {
    ImGui::DragFloat(id.c_str(), &value, sensitivity);
    tooltip(desc);
}
void ImGuiConfiguration::config_float(const std::string& id,
                                      float& value,
                                      const float& min,
                                      const float& max,
                                      const std::string& desc) {
    ImGui::SliderFloat(id.c_str(), &value, min, max);
    tooltip(desc);
}
void ImGuiConfiguration::config_int(const std::string& id, int& value, const std::string& desc) {
    ImGui::DragInt(id.c_str(), &value);
    tooltip(desc);
}
void ImGuiConfiguration::config_int(
    const std::string& id, int& value, const int& min, const int& max, const std::string& desc) {
    ImGui::SliderInt(id.c_str(), &value, min, max);
    tooltip(desc);
}
void ImGuiConfiguration::config_float3(const std::string& id,
                                       float value[3],
                                       const std::string& desc) {
    ImGui::InputFloat3(id.c_str(), value);
    tooltip(desc);
}
void ImGuiConfiguration::config_bool(const std::string& id, bool& value, const std::string& desc) {
    ImGui::Checkbox(id.c_str(), &value);
    tooltip(desc);
}
bool ImGuiConfiguration::config_bool(const std::string& id, const std::string& desc) {
    bool pressed = ImGui::Button(id.c_str());
    tooltip(desc);
    return pressed;
}
void ImGuiConfiguration::config_options(const std::string& id,
                                        int& selected,
                                        const std::vector<std::string>& options,
                                        const OptionsStyle style,
                                        const std::string& desc) {
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
                const std::vector<std::string>* options = reinterpret_cast<std::vector<std::string>*>(data);
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
}
bool ImGuiConfiguration::config_text(const std::string& id,
                                     const uint32_t max_len,
                                     char* string,
                                     const bool needs_submit,
                                     const std::string& desc) {

    bool submit_change = ImGui::InputText(id.c_str(), string, max_len,
                                          needs_submit ? ImGuiInputTextFlags_EnterReturnsTrue : 0);
    tooltip(desc);
    return submit_change;
}

} // namespace merian
