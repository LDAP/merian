#pragma once

#include "imgui.h"
#include "merian/utils/enums.hpp"

#include <array>
#include <string>

namespace merian {

enum class ImGuiTheme {
    Classic, // stock Dear ImGui dark
    Graphite,
    Nord,
    Mocha,
    Carbon,
};

static constexpr std::array<ImGuiTheme, 5> IMGUI_THEME_VALUES = {
    ImGuiTheme::Classic, ImGuiTheme::Graphite, ImGuiTheme::Nord, ImGuiTheme::Mocha,
    ImGuiTheme::Carbon};

template <> inline uint32_t enum_size<ImGuiTheme>() {
    return IMGUI_THEME_VALUES.size();
}
template <> inline const ImGuiTheme* enum_values<ImGuiTheme>() {
    return IMGUI_THEME_VALUES.data();
}
template <> inline std::string enum_to_string<ImGuiTheme>(const ImGuiTheme value) {
    switch (value) {
    case ImGuiTheme::Classic:
        return "Classic";
    case ImGuiTheme::Graphite:
        return "Graphite";
    case ImGuiTheme::Nord:
        return "Nord";
    case ImGuiTheme::Mocha:
        return "Mocha";
    case ImGuiTheme::Carbon:
        return "Carbon";
    }
    return "unknown";
}

// Applies colors, rounding, padding and borders; the font is left untouched.
void apply_imgui_theme(ImGuiStyle& style, ImGuiTheme theme);

// Theme new contexts start with; changing it only affects contexts created afterwards.
ImGuiTheme default_imgui_theme();
void set_default_imgui_theme(ImGuiTheme theme);

} // namespace merian
