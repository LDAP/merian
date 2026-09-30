#include "merian/utils/properties_imgui.hpp"

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace merian {

void tooltip(const std::string& tooltip) {
    if (!tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("%s", tooltip.c_str());
    }
}

static std::string format_for_sensitivity(const float sensitivity) {
    return fmt::format("%.{}f", std::max(0, (int)std::ceil(-std::log10(sensitivity))));
}

template <typename T>
static bool drag_or_slider(const std::string& id,
                           const ImGuiDataType data_type,
                           T* value,
                           const int components,
                           const float sensitivity,
                           const std::optional<T>& min,
                           const std::optional<T>& max,
                           const char* format) {
    if (min && max) {
        const T min_v = *min;
        const T max_v = *max;
        return ImGui::SliderScalarN(id.c_str(), data_type, value, components, &min_v, &max_v,
                                    format);
    }
    const T min_v = min.value_or(T{});
    const T max_v = max.value_or(T{});
    return ImGui::DragScalarN(id.c_str(), data_type, value, components, sensitivity,
                              min ? &min_v : nullptr, max ? &max_v : nullptr, format);
}

ImGuiProperties::~ImGuiProperties() {}

bool ImGuiProperties::st_begin_child(const std::string& id,
                                     const std::string& label,
                                     const ChildFlags flags) {
    if (!open_children.empty() && open_children.back() == OpenChild::TAB_BAR) {
        if (!ImGui::BeginTabItem(label.empty() ? id.c_str() : label.c_str())) {
            return false;
        }
        open_children.push_back(OpenChild::TAB_ITEM);
        return true;
    }
    if ((flags & ChildFlagBits::TABS) != 0) {
        if (!ImGui::BeginTabBar(id.c_str())) {
            return false;
        }
        open_children.push_back(OpenChild::TAB_BAR);
        return true;
    }

    ImGuiTreeNodeFlags imgui_flags{};
    if (flags & ChildFlagBits::DEFAULT_OPEN) {
        imgui_flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
    if (flags & ChildFlagBits::FRAMED) {
        imgui_flags |= ImGuiTreeNodeFlags_Framed;
    }
    if (!ImGui::TreeNodeEx(id.c_str(), imgui_flags, "%s", label.c_str())) {
        return false;
    }
    open_children.push_back(OpenChild::TREE);
    return true;
}
void ImGuiProperties::st_end_child() {
    assert(!open_children.empty());
    const OpenChild open = open_children.back();
    open_children.pop_back();
    switch (open) {
    case OpenChild::TAB_BAR:
        ImGui::EndTabBar();
        break;
    case OpenChild::TAB_ITEM:
        ImGui::EndTabItem();
        break;
    case OpenChild::TREE:
        ImGui::TreePop();
        break;
    }
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

bool ImGuiProperties::config_float(const std::string& id,
                                   float* value,
                                   const std::string& desc,
                                   const int components,
                                   const float sensitivity,
                                   const std::optional<float>& min,
                                   const std::optional<float>& max) {
    const std::string format = format_for_sensitivity(sensitivity);
    const bool value_changed = drag_or_slider<float>(id, ImGuiDataType_Float, value, components,
                                                     sensitivity, min, max, format.c_str());
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_int(const std::string& id,
                                 int32_t* value,
                                 const std::string& desc,
                                 const int components,
                                 const std::optional<int32_t>& min,
                                 const std::optional<int32_t>& max) {
    const bool value_changed =
        drag_or_slider<int32_t>(id, ImGuiDataType_S32, value, components, 1.0f, min, max, nullptr);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_uint(const std::string& id,
                                  uint32_t* value,
                                  const std::string& desc,
                                  const int components,
                                  const std::optional<uint32_t>& min,
                                  const std::optional<uint32_t>& max) {
    const bool value_changed =
        drag_or_slider<uint32_t>(id, ImGuiDataType_U32, value, components, 1.0f, min, max, nullptr);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_uint64(const std::string& id,
                                    uint64_t* value,
                                    const std::string& desc,
                                    const int components,
                                    const std::optional<uint64_t>& min,
                                    const std::optional<uint64_t>& max) {
    const bool value_changed =
        drag_or_slider<uint64_t>(id, ImGuiDataType_U64, value, components, 1.0f, min, max, nullptr);
    tooltip(desc);
    return value_changed;
}

bool ImGuiProperties::config_color3(const std::string& id,
                                    float color[3],
                                    const std::string& desc) {
    const bool value_changed = ImGui::ColorEdit3(id.c_str(), color);
    tooltip(desc);
    return value_changed;
}
bool ImGuiProperties::config_color4(const std::string& id,
                                    float color[4],
                                    const std::string& desc) {
    const bool value_changed = ImGui::ColorEdit4(id.c_str(), color);
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
bool ImGuiProperties::config_split(const std::string& id,
                                   const std::vector<SplitPart>& parts,
                                   const std::string& desc) {
    std::vector<const SplitPart*> shown;
    for (const SplitPart& part : parts) {
        if (part.shown) {
            shown.push_back(&part);
        }
    }
    if (shown.size() < 2) {
        return false;
    }
    const auto count = static_cast<uint32_t>(shown.size());

    bool value_changed = false;
    float total = 0.f;
    for (const SplitPart* part : shown) {
        total += *part->weight;
    }
    if (!(total > 0.f)) {
        for (const SplitPart* part : shown) {
            *part->weight = 1.f;
        }
        total = static_cast<float>(count);
        value_changed = true;
    }

    std::vector<float> ends(count);
    const auto compute_ends = [&] {
        float sum = 0.f;
        for (uint32_t i = 0; i < count; i++) {
            sum += *shown[i]->weight;
            ends[i] = sum / total;
        }
        ends.back() = 1.f;
    };
    compute_ends();

    ImGui::PushID(id.c_str());
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 size{ImGui::CalcItemWidth(), ImGui::GetFrameHeight()};
    ImGui::InvisibleButton("##split", size);
    tooltip(desc);

    const float mouse = std::clamp((ImGui::GetIO().MousePos.x - origin.x) / size.x, 0.f, 1.f);
    ImGuiStorage* storage = ImGui::GetStateStorage();
    const ImGuiID divider_key = ImGui::GetItemID();
    if (ImGui::IsItemActivated()) {
        uint32_t nearest = 0;
        for (uint32_t i = 1; i + 1 < count; i++) {
            const float distance = std::abs(ends[i] - mouse);
            const float nearest_distance = std::abs(ends[nearest] - mouse);
            if (distance < nearest_distance || (distance == nearest_distance && mouse > ends[i])) {
                nearest = i;
            }
        }
        storage->SetInt(divider_key, static_cast<int>(nearest));
    }
    const bool active = ImGui::IsItemActive();
    const auto divider = static_cast<uint32_t>(storage->GetInt(divider_key, 0));
    if (active) {
        const float lo = divider > 0 ? ends[divider - 1] : 0.f;
        const float hi = ends[divider + 1];
        const float at = std::clamp(std::round(mouse * 100.f) / 100.f, lo, hi);
        const float pair = *shown[divider]->weight + *shown[divider + 1]->weight;
        const float left = std::min((at - lo) * total, pair);
        if (left != *shown[divider]->weight) {
            *shown[divider]->weight = left;
            *shown[divider + 1]->weight = pair - left;
            value_changed = true;
            compute_ends();
        }
    }
    if (active || ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 max{origin.x + size.x, origin.y + size.y};
    const float rounding = ImGui::GetStyle().FrameRounding;
    draw_list->AddRectFilled(origin, max, ImGui::GetColorU32(ImGuiCol_FrameBg), rounding);
    float begin = 0.f;
    for (uint32_t i = 0; i < count; i++) {
        const ImVec2 part_min{origin.x + begin * size.x, origin.y};
        const ImVec2 part_max{origin.x + ends[i] * size.x, max.y};
        draw_list->AddRectFilled(
            part_min, part_max,
            ImColor::HSV(static_cast<float>(i) / static_cast<float>(count), 0.45f, 0.5f));

        const std::string text =
            fmt::format("{} {:.0f} %", shown[i]->label, (ends[i] - begin) * 100.f);
        const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
        const ImVec2 text_pos{std::max(part_min.x, (part_min.x + part_max.x - text_size.x) * 0.5f),
                              origin.y + (size.y - text_size.y) * 0.5f};
        draw_list->PushClipRect(part_min, part_max, true);
        draw_list->AddText(text_pos, ImGui::GetColorU32(ImGuiCol_Text), text.c_str());
        draw_list->PopClipRect();
        begin = ends[i];
    }
    for (uint32_t i = 0; i + 1 < count; i++) {
        const float x = origin.x + ends[i] * size.x;
        const ImGuiCol color =
            active && i == divider ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab;
        draw_list->AddLine({x, origin.y}, {x, max.y}, ImGui::GetColorU32(color), 3.f);
    }

    ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
    ImGui::TextUnformatted(id.c_str());
    ImGui::PopID();
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
                                  std::string& string,
                                  const bool needs_submit,
                                  const std::string& desc) {

    bool submit_change = ImGui::InputText(id.c_str(), &string,
                                          needs_submit ? ImGuiInputTextFlags_EnterReturnsTrue : 0);
    tooltip(desc);
    return submit_change;
}

bool ImGuiProperties::config_text_multiline(const std::string& id,
                                            std::string& string,
                                            const bool needs_submit,
                                            const std::string& desc) {
    bool submit_change = ImGui::InputTextMultiline(
        id.c_str(), &string, ImVec2(0, 0), needs_submit ? ImGuiInputTextFlags_EnterReturnsTrue : 0);
    tooltip(desc);
    return submit_change;
}

bool ImGuiProperties::is_ui() {
    return true;
}
bool ImGuiProperties::serialize_json(const std::string&, nlohmann::json&) {
    return false;
}
bool ImGuiProperties::serialize_string(const std::string&, std::string&) {
    return false;
}

} // namespace merian
