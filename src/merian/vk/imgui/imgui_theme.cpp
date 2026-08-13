#include "merian/vk/imgui/imgui_theme.hpp"

#include <algorithm>
#include <cstdint>

namespace merian {

namespace {

ImGuiTheme default_theme = ImGuiTheme::Mocha;

struct Palette {
    ImVec4 bg;         // window background
    ImVec4 surface;    // child windows, frames, popups
    ImVec4 surface_hi; // hovered frames, headers
    ImVec4 border;
    ImVec4 text;
    ImVec4 text_dim;
    ImVec4 accent;
    ImVec4 accent_hi;
};

ImVec4 with_alpha(const ImVec4 c, const float a) {
    return ImVec4(c.x, c.y, c.z, a);
}

ImVec4 lift(const ImVec4 c, const float f) {
    return ImVec4(std::min(c.x + f, 1.f), std::min(c.y + f, 1.f), std::min(c.z + f, 1.f), c.w);
}

ImVec4 rgb(const uint32_t hex, const float a = 1.f) {
    return ImVec4(static_cast<float>((hex >> 16) & 0xFF) / 255.f,
                  static_cast<float>((hex >> 8) & 0xFF) / 255.f,
                  static_cast<float>(hex & 0xFF) / 255.f, a);
}

void apply_palette(ImGuiStyle& style, const Palette& p) {
    style.WindowRounding = 6.f;
    style.ChildRounding = 6.f;
    style.FrameRounding = 4.f;
    style.PopupRounding = 6.f;
    style.GrabRounding = 4.f;
    style.TabRounding = 4.f;
    style.ScrollbarRounding = 8.f;

    style.WindowPadding = ImVec2(10.f, 10.f);
    style.FramePadding = ImVec2(8.f, 4.f);
    style.ItemSpacing = ImVec2(8.f, 5.f);
    style.ItemInnerSpacing = ImVec2(6.f, 4.f);
    style.IndentSpacing = 18.f;
    style.ScrollbarSize = 12.f;
    style.GrabMinSize = 10.f;

    style.WindowBorderSize = 1.f;
    style.ChildBorderSize = 1.f;
    style.FrameBorderSize = 0.f;
    style.PopupBorderSize = 1.f;
    style.TabBorderSize = 0.f;

    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
    style.SeparatorTextBorderSize = 1.f;

    ImVec4* const c = style.Colors;
    c[ImGuiCol_Text] = p.text;
    c[ImGuiCol_TextDisabled] = p.text_dim;
    c[ImGuiCol_WindowBg] = with_alpha(p.bg, 0.98f);
    c[ImGuiCol_ChildBg] = with_alpha(p.surface, 0.35f);
    c[ImGuiCol_PopupBg] = with_alpha(p.bg, 0.99f);
    c[ImGuiCol_Border] = with_alpha(p.border, 0.6f);
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = p.surface;
    c[ImGuiCol_FrameBgHovered] = p.surface_hi;
    c[ImGuiCol_FrameBgActive] = lift(p.surface_hi, 0.04f);
    c[ImGuiCol_TitleBg] = p.bg;
    c[ImGuiCol_TitleBgActive] = p.surface;
    c[ImGuiCol_TitleBgCollapsed] = with_alpha(p.bg, 0.8f);
    c[ImGuiCol_MenuBarBg] = p.surface;
    c[ImGuiCol_ScrollbarBg] = with_alpha(p.bg, 0.4f);
    c[ImGuiCol_ScrollbarGrab] = p.surface_hi;
    c[ImGuiCol_ScrollbarGrabHovered] = lift(p.surface_hi, 0.06f);
    c[ImGuiCol_ScrollbarGrabActive] = with_alpha(p.accent, 0.8f);
    c[ImGuiCol_CheckMark] = p.accent_hi;
    c[ImGuiCol_SliderGrab] = p.accent;
    c[ImGuiCol_SliderGrabActive] = p.accent_hi;
    c[ImGuiCol_Button] = p.surface_hi;
    c[ImGuiCol_ButtonHovered] = with_alpha(p.accent, 0.55f);
    c[ImGuiCol_ButtonActive] = with_alpha(p.accent, 0.8f);
    c[ImGuiCol_Header] = with_alpha(p.accent, 0.22f);
    c[ImGuiCol_HeaderHovered] = with_alpha(p.accent, 0.38f);
    c[ImGuiCol_HeaderActive] = with_alpha(p.accent, 0.5f);
    c[ImGuiCol_Separator] = with_alpha(p.border, 0.5f);
    c[ImGuiCol_SeparatorHovered] = with_alpha(p.accent, 0.6f);
    c[ImGuiCol_SeparatorActive] = p.accent;
    c[ImGuiCol_ResizeGrip] = with_alpha(p.border, 0.4f);
    c[ImGuiCol_ResizeGripHovered] = with_alpha(p.accent, 0.6f);
    c[ImGuiCol_ResizeGripActive] = p.accent;
    c[ImGuiCol_Tab] = p.surface;
    c[ImGuiCol_TabHovered] = with_alpha(p.accent, 0.45f);
    c[ImGuiCol_TabSelected] = with_alpha(p.accent, 0.3f);
    c[ImGuiCol_TabDimmed] = p.bg;
    c[ImGuiCol_TabDimmedSelected] = p.surface;
    c[ImGuiCol_PlotLines] = p.accent_hi;
    c[ImGuiCol_PlotLinesHovered] = lift(p.accent_hi, 0.1f);
    c[ImGuiCol_PlotHistogram] = p.accent;
    c[ImGuiCol_PlotHistogramHovered] = p.accent_hi;
    c[ImGuiCol_TableHeaderBg] = p.surface;
    c[ImGuiCol_TableBorderStrong] = with_alpha(p.border, 0.8f);
    c[ImGuiCol_TableBorderLight] = with_alpha(p.border, 0.4f);
    c[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = with_alpha(p.surface, 0.3f);
    c[ImGuiCol_TextSelectedBg] = with_alpha(p.accent, 0.35f);
    c[ImGuiCol_DragDropTarget] = p.accent_hi;
    c[ImGuiCol_NavCursor] = with_alpha(p.accent, 0.8f);
    c[ImGuiCol_NavWindowingHighlight] = with_alpha(p.accent, 0.7f);
    c[ImGuiCol_NavWindowingDimBg] = ImVec4(0.1f, 0.1f, 0.1f, 0.5f);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.6f);
}

} // namespace

void apply_imgui_theme(ImGuiStyle& style, const ImGuiTheme theme) {
    style = ImGuiStyle(); // reset paddings/roundings the previous theme may have set
    switch (theme) {
    case ImGuiTheme::Classic:
        ImGui::StyleColorsDark(&style);
        return;
    case ImGuiTheme::Graphite:
        apply_palette(style, Palette{
                                 .bg = rgb(0x141518),
                                 .surface = rgb(0x1f2126),
                                 .surface_hi = rgb(0x2a2d33),
                                 .border = rgb(0x3a3d45),
                                 .text = rgb(0xdcdee3),
                                 .text_dim = rgb(0x84878f),
                                 .accent = rgb(0x4f8cc9),
                                 .accent_hi = rgb(0x76aede),
                             });
        return;
    case ImGuiTheme::Nord:
        apply_palette(style, Palette{
                                 .bg = rgb(0x2e3440),
                                 .surface = rgb(0x3b4252),
                                 .surface_hi = rgb(0x434c5e),
                                 .border = rgb(0x4c566a),
                                 .text = rgb(0xe5e9f0),
                                 .text_dim = rgb(0x9099a8),
                                 .accent = rgb(0x88c0d0),
                                 .accent_hi = rgb(0x8fbcbb),
                             });
        return;
    case ImGuiTheme::Mocha:
        apply_palette(style, Palette{
                                 .bg = rgb(0x1e1e2e),
                                 .surface = rgb(0x313244),
                                 .surface_hi = rgb(0x45475a),
                                 .border = rgb(0x585b70),
                                 .text = rgb(0xcdd6f4),
                                 .text_dim = rgb(0x8f93ab),
                                 .accent = rgb(0xcba6f7),
                                 .accent_hi = rgb(0xb4befe),
                             });
        return;
    case ImGuiTheme::Carbon:
        apply_palette(style, Palette{
                                 .bg = rgb(0x161616),
                                 .surface = rgb(0x262626),
                                 .surface_hi = rgb(0x333333),
                                 .border = rgb(0x3d3d3d),
                                 .text = rgb(0xf4f4f4),
                                 .text_dim = rgb(0x8d8d8d),
                                 .accent = rgb(0x78a9ff),
                                 .accent_hi = rgb(0xa6c8ff),
                             });
        return;
    }
}

ImGuiTheme default_imgui_theme() {
    return default_theme;
}

void set_default_imgui_theme(const ImGuiTheme theme) {
    default_theme = theme;
}

} // namespace merian
