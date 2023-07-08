#include "configuration_imgui.hpp"
#include "imgui.h"
#include <algorithm>
#include <fmt/format.h>
#include <stdexcept>

namespace merian {

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
    ImGui::Text("%s", text.c_str());
}
void ImGuiConfiguration::output_plot_line(const std::string& label,
                                          const std::vector<float>& samples,
                                          const float scale_min,
                                          const float scale_max) {
    ImGui::PlotLines(label.c_str(), samples.data(), samples.size(), 0, NULL, scale_min, scale_max);
}

void ImGuiConfiguration::config_color(const std::string& id, glm::vec3& color) {
    ImGui::ColorEdit3(id.c_str(), &color.x);
}
void ImGuiConfiguration::config_color(const std::string& id, glm::vec4& color) {
    ImGui::ColorEdit4(id.c_str(), &color.x);
}
void ImGuiConfiguration::config_vec(const std::string& id, glm::vec3& value) {
    ImGui::InputFloat3(id.c_str(), &value.x);
}
void ImGuiConfiguration::config_vec(const std::string& id, glm::vec4& value) {
    ImGui::InputFloat4(id.c_str(), &value.x);
}
void ImGuiConfiguration::config_angle(const std::string& id, float& angle) {
    ImGui::SliderAngle(id.c_str(), &angle);
}
void ImGuiConfiguration::config_percent(const std::string& id, float& value) {
    ImGui::SliderFloat(id.c_str(), &value, 0, 1, "%.06f");
}
void ImGuiConfiguration::config_float(const std::string& id, float& value) {
    ImGui::DragFloat(id.c_str(), &value);
}
void ImGuiConfiguration::config_float(const std::string& id,
                                      float& value,
                                      const float& min,
                                      const float& max) {
    ImGui::SliderFloat(id.c_str(), &value, min, max);
}
void ImGuiConfiguration::config_int(const std::string& id, int& value) {
    ImGui::DragInt(id.c_str(), &value);
}
void ImGuiConfiguration::config_int(const std::string& id,
                                    int& value,
                                    const int& min,
                                    const int& max) {
    ImGui::SliderInt(id.c_str(), &value, min, max);
}
void ImGuiConfiguration::config_float3(const std::string& id, float value[3]) {
    ImGui::InputFloat3(id.c_str(), value);
}
void ImGuiConfiguration::config_bool(const std::string& id, bool& value) {
    ImGui::Checkbox(id.c_str(), &value);
}
bool ImGuiConfiguration::config_bool(const std::string& id) {
    return ImGui::Button(id.c_str());
}
void ImGuiConfiguration::config_options(const std::string& id,
                                        int& selected,
                                        const std::vector<std::string>& options,
                                        const OptionsStyle style) {
    switch (style) {
    case OptionsStyle::RADIO_BUTTON:
        for (uint32_t i = 0; i < options.size(); i++) {
            ImGui::RadioButton(options[i].c_str(), &selected, i);
        }
        break;
    case OptionsStyle::COMBO:
        ImGui::Combo(id.c_str(), &selected, fmt::format("{}", fmt::join(options, "\0")).c_str());
        break;
    case OptionsStyle::DONT_CARE:
    case OptionsStyle::LIST_BOX: {
        std::vector<const char*> options_c_str;
        std::transform(options.begin(), options.end(), std::back_inserter(options_c_str),
                       [](auto& str) { return str.c_str(); });
        ImGui::ListBox(id.c_str(), &selected, options_c_str.data(), options_c_str.size());
        break;
    }
    default:
        throw std::runtime_error{"OptionsStyle not supported"};
    }
}
bool ImGuiConfiguration::config_text(const std::string& id,
                                     const uint32_t max_len,
                                     char* string,
                                     const bool needs_submit) {

    return ImGui::InputText(id.c_str(), string, max_len,
                            needs_submit ? ImGuiInputTextFlags_EnterReturnsTrue : 0);
}

} // namespace merian
