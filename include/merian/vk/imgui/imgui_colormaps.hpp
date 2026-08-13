#pragma once

#include "imgui.h"

namespace merian {

// Host mirrors of merian-shaders/colors/colormaps.slang, for legends and plots.

// Polynomial fit of the Turbo colormap (Mikhailov, Google AI Blog 2019).
ImU32 imgui_colormap_turbo(float x);

// Diverging blue-white-red map for signed quantities; 0.5 = neutral.
ImU32 imgui_colormap_diverging(float x);

// Draws a horizontal colorbar into the draw list.
void imgui_colorbar(ImDrawList* dl, ImVec2 min, ImVec2 max, ImU32 (*colormap)(float));

} // namespace merian
