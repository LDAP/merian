#include "merian-graph/nodes/path_debug/path_debug.hpp"

#include "merian-shaders/debug/path_record.hpp"
#include "merian/utils/colors.hpp"
#include "merian/utils/properties_imgui.hpp"
#include "merian/vk/imgui/imgui_colormaps.hpp"

#include <fmt/format.h>
#include <imgui.h>

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

namespace merian {

namespace {

using Stats = std::array<uint32_t, PATH_DEBUG_STATS_UINTS>;

float stat_float(const Stats& stats, const uint32_t slot) {
    return std::bit_cast<float>(stats[slot]);
}

float3 stat_rgb(const Stats& stats, const uint32_t slot) {
    return {stat_float(stats, slot), stat_float(stats, slot + 1), stat_float(stats, slot + 2)};
}

float total_variation_percent(const Stats& stats, const uint32_t slot) {
    return std::min(0.5f * stat_float(stats, slot), 1.f) * 100.f;
}

float3 check_albedo(const Stats& stats) {
    return stat_rgb(stats, PATH_DEBUG_CHECK_WEIGHT) /
           static_cast<float>(std::max(stats[PATH_DEBUG_CHECK_DRAWS], 1u));
}

std::string elided(std::string text, const float width) {
    if (ImGui::CalcTextSize(text.c_str()).x <= width) {
        return text;
    }
    while (!text.empty() && ImGui::CalcTextSize((text + "...").c_str()).x > width) {
        text.pop_back();
    }
    return text + "...";
}

constexpr ImVec4 GOOD_COLOR{0.35f, 0.85f, 0.4f, 1.f};
constexpr ImVec4 WARN_COLOR{0.95f, 0.75f, 0.3f, 1.f};
constexpr ImVec4 BAD_COLOR{0.95f, 0.35f, 0.3f, 1.f};
constexpr ImU32 LABEL_COLOR = IM_COL32(235, 235, 235, 255);
constexpr ImU32 RECORDED_COLOR = IM_COL32(203, 166, 247, 255);

ImVec4 to_imvec4(const float3& rgb) {
    return {rgb.x, rgb.y, rgb.z, 1.f};
}

void shadowed_text(ImDrawList* const dl,
                   const ImVec2 pos,
                   const char* text,
                   const ImU32 color = LABEL_COLOR) {
    dl->AddText(ImVec2(pos.x + 1.f, pos.y + 1.f), IM_COL32(0, 0, 0, 200), text);
    dl->AddText(pos, color, text);
}

// chi2/dof of a correct sampler is 1 with a standard deviation of sqrt(2 / dof)
float chi2_sigma(const uint32_t dof) {
    return std::sqrt(2.f / static_cast<float>(std::max(dof, 1u)));
}

bool per_steradian(const uint32_t channel) {
    return channel != PATH_DEBUG_MAP_COVERAGE && channel != PATH_DEBUG_MAP_RADIANCE_COLOR &&
           channel != PATH_DEBUG_MAP_MEAN && channel != PATH_DEBUG_MAP_MEAN_COLOR;
}

bool colored(const uint32_t channel) {
    return channel == PATH_DEBUG_MAP_INTEGRAND_COLOR || channel == PATH_DEBUG_MAP_RADIANCE_COLOR ||
           channel == PATH_DEBUG_MAP_MEAN_COLOR;
}

bool diverging(const uint32_t channel) {
    return channel == PATH_DEBUG_MAP_RATIO || channel == PATH_DEBUG_MAP_ZSCORE ||
           channel == PATH_DEBUG_MAP_COVERAGE;
}

float reduced_chi2(const Stats& stats) {
    return stat_float(stats, PATH_DEBUG_STATS_CHI2) /
           static_cast<float>(std::max(stats[PATH_DEBUG_STATS_CHI2_DOF], 1u));
}

ImVec4 chi2_color(const float reduced, const uint32_t dof) {
    const float deviation = std::abs(reduced - 1.f) / chi2_sigma(dof);
    if (deviation < 3.f) {
        return GOOD_COLOR;
    }
    if (deviation < 5.f) {
        return WARN_COLOR;
    }
    return BAD_COLOR;
}

std::string si_count(const uint64_t n) {
    if (n >= 1'000'000'000) {
        return fmt::format("{:.1f}G", static_cast<double>(n) / 1e9);
    }
    if (n >= 1'000'000) {
        return fmt::format("{:.1f}M", static_cast<double>(n) / 1e6);
    }
    if (n >= 10'000) {
        return fmt::format("{:.0f}k", static_cast<double>(n) / 1e3);
    }
    return fmt::format("{}", n);
}

void wrapped_text(const std::string& text) {
    ImGui::PushTextWrapPos(0.f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
}

bool direction_disc(const char* id, float& theta_deg, float& phi_deg) {
    constexpr float RADIUS = 76.f;
    constexpr float PAD = 4.f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 center(origin.x + RADIUS + PAD, origin.y + RADIUS + PAD);
    ImGui::InvisibleButton(id, ImVec2(2.f * (RADIUS + PAD), 2.f * (RADIUS + PAD)));
    const bool below = theta_deg > 90.f;

    bool changed = false;
    if (ImGui::IsItemActive()) {
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const float dx = (mouse.x - center.x) / RADIUS;
        const float dy = (mouse.y - center.y) / RADIUS;
        const float r = std::sqrt((dx * dx) + (dy * dy));
        if (r > 1e-3f) {
            phi_deg = degrees(std::atan2(-dy, dx));
            const float polar = std::min(r, 1.f) * 90.f;
            theta_deg = below ? 180.f - polar : polar;
            changed = true;
        }
    }

    ImDrawList* const dl = ImGui::GetWindowDrawList();
    dl->AddCircleFilled(center, RADIUS, IM_COL32(28, 28, 32, 255), 64);
    for (uint32_t ring = 1; ring <= 3; ring++) {
        dl->AddCircle(center, RADIUS * static_cast<float>(ring) / 3.f, IM_COL32(90, 90, 96, 255),
                      64);
    }
    dl->AddLine(ImVec2(center.x - RADIUS, center.y), ImVec2(center.x + RADIUS, center.y),
                IM_COL32(70, 70, 76, 255));
    dl->AddLine(ImVec2(center.x, center.y - RADIUS), ImVec2(center.x, center.y + RADIUS),
                IM_COL32(70, 70, 76, 255));
    shadowed_text(dl, ImVec2(center.x + RADIUS - 24.f, center.y - 16.f), "0°");
    shadowed_text(dl, ImVec2(center.x + 4.f, center.y - RADIUS - 2.f), "90°");

    const float polar = below ? 180.f - theta_deg : theta_deg;
    const float rad = radians(phi_deg);
    const float handle_r = RADIUS * std::clamp(polar / 90.f, 0.f, 1.f);
    const ImVec2 handle(center.x + (handle_r * std::cos(rad)),
                        center.y - (handle_r * std::sin(rad)));
    dl->AddLine(center, handle, IM_COL32(200, 200, 90, 180), 1.5f);
    if (below) {
        dl->AddCircle(handle, 6.f, IM_COL32(120, 170, 255, 255), 16, 2.f);
    } else {
        dl->AddCircleFilled(handle, 5.f, IM_COL32(240, 220, 90, 255), 16);
    }
    return changed;
}

struct BSDFVerdict {
    enum class Status { OK, FAIL, APPROXIMATE, NO_STATEMENT };
    const char* label;
    Status status;
    std::string detail;
};

BSDFVerdict::Status passed(const bool ok) {
    return ok ? BSDFVerdict::Status::OK : BSDFVerdict::Status::FAIL;
}

void check_row(const BSDFVerdict& verdict) {
    ImGui::TextUnformatted(verdict.label);
    ImGui::SameLine(210.f);
    switch (verdict.status) {
    case BSDFVerdict::Status::OK:
        ImGui::TextColored(GOOD_COLOR, "OK (%s)", verdict.detail.c_str());
        break;
    case BSDFVerdict::Status::FAIL:
        ImGui::TextColored(BAD_COLOR, "FAIL (%s)", verdict.detail.c_str());
        break;
    case BSDFVerdict::Status::APPROXIMATE:
        ImGui::TextColored(WARN_COLOR, "APPROX (%s)", verdict.detail.c_str());
        break;
    case BSDFVerdict::Status::NO_STATEMENT:
        ImGui::TextDisabled("n/a (%s)", verdict.detail.c_str());
        break;
    }
}

std::vector<BSDFVerdict> bsdf_verdicts(const Stats& stats) {
    const uint32_t draws = stats[PATH_DEBUG_CHECK_DRAWS];
    if (draws == 0 || stats[PATH_DEBUG_CHECK_CELLS] == 0) {
        return {};
    }

    const float3 albedo_mc = check_albedo(stats);
    const float albedo_lum = yuv_luminance(albedo_mc);
    const float albedo_max = std::max({albedo_mc.x, albedo_mc.y, albedo_mc.z});
    const float albedo_stderr =
        std::sqrt(stat_float(stats, PATH_DEBUG_CHECK_WEIGHT_M2)) / static_cast<float>(draws);
    const uint32_t transmitted = stats[PATH_DEBUG_CHECK_TRANSMITTED];
    const uint32_t bad = stats[PATH_DEBUG_CHECK_BAD];
    const uint32_t bad_cells = stats[PATH_DEBUG_CHECK_BAD_CELLS];
    const bool resolved = stat_float(stats, PATH_DEBUG_CHECK_ALPHA) >= 0.0225f;
    const char* const unresolved = "lobe narrower than the grid";
    const auto identity = [&](const char* label, const uint32_t off_slot, const uint32_t count,
                              const float worst) {
        const float share = static_cast<float>(stats[off_slot]) / static_cast<float>(count);
        return BSDFVerdict{label, passed(share <= 1e-3f),
                           fmt::format("{:.3f}% off, worst {:.0e}", share * 100.f, worst)};
    };
    const auto integral = [&](const char* label, const bool ok, const std::string& detail) {
        if (!resolved && !ok) {
            return BSDFVerdict{label, BSDFVerdict::Status::NO_STATEMENT,
                               fmt::format("{}, {}", unresolved, detail)};
        }
        return BSDFVerdict{label, passed(ok), detail};
    };

    std::vector<BSDFVerdict> verdicts;
    const float mc_tol = std::max(6.f * albedo_stderr, 1e-3f);
    const bool energy_ok = albedo_max <= 1.f + mc_tol;
    verdicts.push_back({"energy conservation", passed(energy_ok),
                        energy_ok
                            ? fmt::format("albedo {:.4f} ± {:.4f}", albedo_max, albedo_stderr)
                            : fmt::format("albedo {:.4f}{}", albedo_max,
                                          transmitted > 0 ? ", refraction scales by eta²" : "")});

    verdicts.push_back(identity("reciprocity", PATH_DEBUG_CHECK_OFF_RECIPROCITY,
                                std::max(stats[PATH_DEBUG_CHECK_RECIPROCITY_CELLS], 1u),
                                stat_float(stats, PATH_DEBUG_CHECK_ERR_RECIPROCITY)));

    // a draw sample() rejects carries no density, so the pdf integrates to the valid share
    const float pdf_integral = stat_float(stats, PATH_DEBUG_CHECK_PDF_INTEGRAL);
    const float valid_share =
        static_cast<float>(stats[PATH_DEBUG_CHECK_VALID]) / static_cast<float>(draws);
    verdicts.push_back(
        integral("pdf normalized", std::abs(pdf_integral - valid_share) <= 1e-2f,
                 fmt::format("∫pdf dω {:.4f} vs valid draws {:.4f}", pdf_integral, valid_share)));

    verdicts.push_back(identity("sample weight = eval / pdf", PATH_DEBUG_CHECK_OFF_WEIGHT, draws,
                                stat_float(stats, PATH_DEBUG_CHECK_ERR_WEIGHT)));
    verdicts.push_back(identity("sampled pdf = pdf()", PATH_DEBUG_CHECK_OFF_PDF, draws,
                                stat_float(stats, PATH_DEBUG_CHECK_ERR_PDF)));

    const bool finite_ok = bad == 0 && bad_cells == 0;
    verdicts.push_back(
        {"finite and non-negative", passed(finite_ok),
         finite_ok ? fmt::format("{} samples", si_count(draws))
                   : fmt::format("{} samples, {} cells", si_count(bad), si_count(bad_cells))});

    const float eval_lum = yuv_luminance(stat_rgb(stats, PATH_DEBUG_CHECK_EVAL_INTEGRAL));
    const float eval_rel =
        std::abs(eval_lum - albedo_lum) / std::max({eval_lum, albedo_lum, 1e-6f});
    verdicts.push_back(
        integral("sampling integrates eval", eval_rel <= 0.02f,
                 fmt::format("∫f·cos dω {:.4f} vs sampled {:.4f}", eval_lum, albedo_lum)));

    // an approximation for demodulation, without the eta^2 radiance scaling of refraction
    const float albedo_fn = yuv_luminance(stat_rgb(stats, PATH_DEBUG_CHECK_ALBEDO));
    const float albedo_rel = (albedo_fn - albedo_lum) / std::max({albedo_fn, albedo_lum, 1e-6f});
    const float refracted =
        stat_float(stats, PATH_DEBUG_CHECK_TRANSMITTED_WEIGHT) / static_cast<float>(draws);
    const std::string albedo_detail =
        fmt::format("{:.4f} vs {:.4f}, {:+.0f}%", albedo_fn, albedo_lum, albedo_rel * 100.f);
    if (std::abs(albedo_rel) <= 0.15f) {
        verdicts.push_back({"get_albedo()", BSDFVerdict::Status::OK, albedo_detail});
    } else if (refracted > 0.f) {
        verdicts.push_back({"get_albedo()", BSDFVerdict::Status::NO_STATEMENT,
                            fmt::format("{:.4f} vs {:.4f}, {:.4f} refracted carries eta²",
                                        albedo_fn, albedo_lum, refracted)});
    } else {
        verdicts.push_back({"get_albedo()", BSDFVerdict::Status::APPROXIMATE, albedo_detail});
    }
    return verdicts;
}

void vertex_row(const uint32_t index, const uint32_t* const words) {
    const auto word_f = [&](const uint32_t offset) { return std::bit_cast<float>(words[offset]); };
    const uint32_t meta = words[PATH_RECORD_VERTEX_META];
    const bool terminal = (meta & PATH_RECORD_FLAG_TERMINAL) != 0;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Text("%u", index);
    ImGui::TableNextColumn();
    const std::string event =
        terminal ? "end"
        : (meta & PATH_RECORD_FLAG_NEE) != 0
            ? "NEE"
            : path_event_name((meta >> PATH_RECORD_LOBE_SHIFT) & PATH_RECORD_LOBE_MASK);
    ImGui::TextUnformatted(event.c_str());
    ImGui::TableNextColumn();
    const float pdf = word_f(PATH_RECORD_VERTEX_PDF);
    const bool resampled = (meta & PATH_RECORD_FLAG_RESAMPLED) != 0;
    if (resampled && !(pdf > 0.f)) {
        ImGui::TextDisabled("-");
    } else if ((meta & PATH_RECORD_FLAG_NEE) != 0 && !resampled && !(pdf > 0.f)) {
        ImGui::TextDisabled("no sample");
    } else if ((meta & (PATH_RECORD_FLAG_REUSED | PATH_RECORD_FLAG_RESAMPLED)) != 0) {
        ImGui::Text("1/W %.4g", pdf);
    } else {
        ImGui::Text("%.4g", pdf);
    }
    ImGui::TableNextColumn();
    ImGui::Text("%.4g", word_f(PATH_RECORD_VERTEX_THROUGHPUT));
    ImGui::TableNextColumn();
    const float3 contrib(word_f(PATH_RECORD_VERTEX_CONTRIB + 0),
                         word_f(PATH_RECORD_VERTEX_CONTRIB + 1),
                         word_f(PATH_RECORD_VERTEX_CONTRIB + 2));
    const float contrib_lum = yuv_luminance(contrib);
    if (std::isfinite(contrib_lum)) {
        ImGui::Text("%.3g", contrib_lum);
    } else {
        ImGui::TextColored(BAD_COLOR, "%.3g, %.3g, %.3g", contrib.x, contrib.y, contrib.z);
    }
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(terminal ? ""
                                    : path_record_method_name((meta >> PATH_RECORD_METHOD_SHIFT) &
                                                              PATH_RECORD_METHOD_MASK));
    ImGui::TableNextColumn();
    const uint32_t material = (meta >> PATH_RECORD_MATERIAL_SHIFT) & PATH_RECORD_MATERIAL_MASK;
    if (!terminal && material != PATH_RECORD_MATERIAL_NONE) {
        ImGui::Text("%u", material);
    }
    ImGui::TableNextColumn();
    const float3 pos(word_f(PATH_RECORD_VERTEX_POS + 0), word_f(PATH_RECORD_VERTEX_POS + 1),
                     word_f(PATH_RECORD_VERTEX_POS + 2));
    if (length(pos) >= PATH_DEBUG_ENV_DISTANCE) {
        ImGui::TextDisabled("environment");
    } else {
        ImGui::Text("%.2f %.2f %.2f", pos.x, pos.y, pos.z);
    }
}

// one button per option, the chosen one highlighted
template <std::size_t N>
bool segmented(const char* id, int32_t& value, const std::array<const char*, N>& names) {
    ImGui::PushID(id);
    bool changed = false;
    for (std::size_t i = 0; i < N; i++) {
        if (i > 0) {
            ImGui::SameLine(0.f, 1.f);
        }
        const bool chosen = value == static_cast<int32_t>(i);
        ImGui::PushStyleColor(
            ImGuiCol_Button,
            ImGui::GetStyleColorVec4(chosen ? ImGuiCol_TabSelected : ImGuiCol_FrameBg));
        if (ImGui::SmallButton(names[i]) && !chosen) {
            value = static_cast<int32_t>(i);
            changed = true;
        }
        ImGui::PopStyleColor();
    }
    ImGui::PopID();
    return changed;
}

// a label, then a combo over the names
template <std::size_t N>
void labeled_combo(const char* label,
                   int32_t& value,
                   const std::array<const char*, N>& names,
                   const float width) {
    ImGui::PushID(label);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(width);
    if (ImGui::BeginCombo("##combo", names.at(value))) {
        for (int32_t i = 0; i < static_cast<int32_t>(N); i++) {
            if (ImGui::Selectable(names.at(i), i == value)) {
                value = i;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopID();
}

} // namespace

void PathDebugNode::draw_window() {
    const float font = ImGui::GetFontSize();
    const ImVec2 position(440.f, 40.f);
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(position, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(
        ImVec2(std::min(43.f * font, display.x - position.x),
               std::min(38.f * ImGui::GetFrameHeightWithSpacing(), display.y - (2.f * position.y))),
        ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Path Debugger")) {
        ImGui::End();
        return;
    }

    window_drawn = true;
    draw_header();

    // the header stays, the tab scrolls
    const auto tab = [this, font](const char* label, void (PathDebugNode::*content)()) {
        if (!ImGui::BeginTabItem(label)) {
            return;
        }
        if (ImGui::BeginChild("##content")) {
            ImGui::PushItemWidth(-10.f * font);
            (this->*content)();
            ImGui::PopItemWidth();
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    };
    if (ImGui::BeginTabBar("tabs")) {
        tab("Paths", &PathDebugNode::draw_paths_tab);
        tab("View", &PathDebugNode::draw_view_tab);
        tab("Analysis", &PathDebugNode::draw_analysis_tab);
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void PathDebugNode::draw_image_choice() {
    const float margin = ImGui::GetStyle().WindowPadding.x;
    ImGui::SetNextWindowPos(ImVec2(margin, ImGui::GetIO().DisplaySize.y - margin), ImGuiCond_Always,
                            ImVec2(0.f, 1.f));
    ImGui::SetNextWindowBgAlpha(0.75f);
    constexpr ImGuiWindowFlags FLAGS =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
    if (ImGui::Begin("##image choice", nullptr, FLAGS)) {
        labeled_combo("image", backdrop, BACKDROP_NAMES, 13.f * ImGui::GetFontSize());
        if (backdrop_missing) {
            ImGui::SameLine();
            ImGui::TextColored(WARN_COLOR, "%s",
                               shows_reference() ? "reference required" : "input required");
        }
    }
    ImGui::End();
}

void PathDebugNode::draw_properties_section(const char* id,
                                            void (PathDebugNode::*section)(Properties&)) {
    ImGui::PushID(id);
    ImGuiProperties config;
    (this->*section)(config);
    ImGui::PopID();
}

void PathDebugNode::draw_header() {
    const float font = ImGui::GetFontSize();
    const ImGuiStyle& style = ImGui::GetStyle();
    const auto right_aligned = [&](const std::initializer_list<const char*> buttons) {
        float width = 0.f;
        for (const char* button : buttons) {
            width +=
                ImGui::CalcTextSize(button).x + (2.f * style.FramePadding.x) + style.ItemSpacing.x;
        }
        ImGui::SameLine(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x -
                                                             width + style.ItemSpacing.x));
    };

    // 1. capture state, path counts and the session settings
    const int32_t state = capture;
    constexpr std::array<const char*, 3> BADGES = {"LIVE###state", "FROZEN###state", "OFF###state"};
    constexpr std::array<ImVec4, 3> BADGE_COLORS = {ImVec4(0.42f, 0.85f, 0.45f, 1.f),
                                                    ImVec4(1.f, 0.76f, 0.25f, 1.f),
                                                    ImVec4(0.6f, 0.6f, 0.62f, 1.f)};
    const ImVec4& color = BADGE_COLORS.at(state);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(color.x, color.y, color.z, 0.12f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(color.x, color.y, color.z, 0.28f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(color.x, color.y, color.z, 0.42f));
    if (ImGui::SmallButton(BADGES.at(state))) {
        capture = (state + 1) % static_cast<int32_t>(CAPTURE_STATE_NAMES.size());
    }
    ImGui::PopStyleColor(4);
    ImGui::SetItemTooltip("%s", state == CAPTURE_LIVE ? "Click to freeze the recorded paths"
                                : state == CAPTURE_FROZEN
                                    ? "Click to turn off recording, analysis and drawing"
                                    : "Click to record again");
    ImGui::SameLine();
    if (capture == CAPTURE_OFF) {
        ImGui::TextDisabled("not recording");
    } else if (readback_valid) {
        ImGui::TextDisabled("%u / %s", latest_readback.frame.draw_count,
                            si_count(latest_readback.frame.paths).c_str());
        ImGui::SetItemTooltip(
            "drawn / recorded paths\n%s of %s record slots · record "
            "probability %.3g\n%s match the selection%s",
            si_count(latest_readback.frame.paths).c_str(), si_count(record_path_capacity).c_str(),
            params.keep_prob, si_count(latest_readback.frame.matched).c_str(),
            capture == CAPTURE_FROZEN
                ? fmt::format("\nheld from frame {}", latest_readback.frame.records_frame).c_str()
                : "");
    }
    struct Popup {
        const char* label;
        const char* id;
        void (PathDebugNode::*content)(Properties&);
    };
    const std::array<Popup, 3> popups = {
        Popup{"Capture", "capture", &PathDebugNode::properties_capture},
        Popup{"Export", "export", &PathDebugNode::properties_export},
        Popup{"Keys", "keys", &PathDebugNode::properties_keys}};
    right_aligned({popups[0].label, popups[1].label, popups[2].label});
    for (const Popup& popup : popups) {
        if (ImGui::SmallButton(popup.label)) {
            ImGui::OpenPopup(popup.label);
        }
        ImGui::SameLine();
    }
    ImGui::NewLine();
    ImGui::PushItemWidth(-8.f * font);
    for (const Popup& popup : popups) {
        if (ImGui::BeginPopup(popup.label)) {
            draw_properties_section(popup.id, popup.content);
            ImGui::EndPopup();
        }
    }
    ImGui::PopItemWidth();

    // 2. the selection
    int32_t kind = select_kind;
    if (segmented("kind", kind, SELECT_NAMES)) {
        set_select_kind(kind);
    }
    if (select_kind == SELECT_REGION) {
        ImGui::SameLine();
        int32_t match = region_match;
        if (segmented("match", match, MATCH_NAMES)) {
            set_region_match(match);
        }
    }
    ImGui::SameLine();
    const std::string status = selection_status();
    if (has_selection && (selected_offscreen || point_hidden())) {
        ImGui::TextColored(WARN_COLOR, "%s", status.c_str());
    } else if (!has_selection || !selected_pos_valid) {
        ImGui::TextDisabled("%s", status.c_str());
    } else {
        ImGui::TextUnformatted(status.c_str());
    }
    right_aligned({"Clear"});
    ImGui::BeginDisabled(!has_selection);
    if (ImGui::SmallButton("Clear")) {
        clear_selection();
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("%s+Esc", modifier_name());

    // 3. the path color
    labeled_combo("color", overlay_color, COLOR_NAMES, 13.f * font);
    ImGui::SameLine();
    draw_color_legend();
    ImGui::Separator();
}

std::string PathDebugNode::selection_status() const {
    if (!has_selection) {
        return fmt::format("no selection · {}+click the image", modifier_name());
    }
    if (selected_offscreen) {
        return "off screen";
    }
    if (!selected_pos_valid) {
        return "resolving surface";
    }
    if (point_hidden()) {
        return "point occluded · no paths";
    }
    if (select_kind == SELECT_POINT) {
        return fmt::format("pixel {}, {}", selected_pixel.x, selected_pixel.y);
    }
    return fmt::format("radius {:.3g}", selected_radius);
}

void PathDebugNode::draw_color_legend() const {
    const float scale = std::max(overlay_exposure, 1e-12f);
    const auto range = [scale](const float log2_value) {
        return fmt::format("{:.3g}", std::exp2(log2_value) / scale);
    };
    const auto bar = [](ImU32 (*colormap)(float), const std::string& low, const std::string& high) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", low.c_str());
        ImGui::SameLine();
        const float line = ImGui::GetFrameHeight();
        const float width =
            std::max(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(high.c_str()).x -
                         ImGui::GetStyle().ItemSpacing.x,
                     40.f);
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        imgui_colorbar(ImGui::GetWindowDrawList(), ImVec2(origin.x, origin.y + (0.3f * line)),
                       ImVec2(origin.x + width, origin.y + (0.7f * line)), colormap);
        ImGui::Dummy(ImVec2(width, line));
        ImGui::SameLine();
        ImGui::TextDisabled("%s", high.c_str());
    };
    switch (overlay_color) {
    case PATH_DEBUG_COLOR_LUMINANCE:
        bar(imgui_colormap_turbo, range(-0.5f * PATH_DEBUG_LUMINANCE_STOPS),
            range(0.5f * PATH_DEBUG_LUMINANCE_STOPS));
        break;
    case PATH_DEBUG_COLOR_CONTRIBUTION:
        bar(imgui_colormap_plasma, range(-0.5f * PATH_DEBUG_CONTRIBUTION_STOPS),
            range(0.5f * PATH_DEBUG_CONTRIBUTION_STOPS));
        break;
    case PATH_DEBUG_COLOR_THROUGHPUT_GAIN:
        bar(imgui_colormap_diverging,
            fmt::format("{:.3g}", std::exp2(-0.5f * PATH_DEBUG_GAIN_STOPS)),
            fmt::format("{:.3g}", std::exp2(0.5f * PATH_DEBUG_GAIN_STOPS)));
        break;
    case PATH_DEBUG_COLOR_SCATTER_EVENTS:
        bar(imgui_colormap_turbo, "0",
            fmt::format("{} scatter events", PATH_DEBUG_SCATTER_COLOR_EVENTS));
        break;
    case PATH_DEBUG_COLOR_TECHNIQUE:
        ImGui::AlignTextToFramePadding();
        for (const uint32_t method :
             {PATH_RECORD_METHOD_BSDF, PATH_RECORD_METHOD_GUIDING, PATH_RECORD_METHOD_LIGHT,
              PATH_RECORD_METHOD_REPLAYED_BSDF, PATH_RECORD_METHOD_REPLAYED_LIGHT,
              PATH_RECORD_METHOD_RECONNECTION, PATH_RECORD_METHOD_UNKNOWN}) {
            ImGui::TextColored(
                to_imvec4(path_debug_unpack_rgb(PATH_DEBUG_METHOD_COLORS[method])), "%s",
                method == PATH_RECORD_METHOD_UNKNOWN ? "other" : PATH_RECORD_METHOD_NAMES[method]);
            ImGui::SameLine();
        }
        ImGui::NewLine();
        break;
    default:
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("one color per path");
        break;
    }
}

void PathDebugNode::draw_paths_tab() {
    draw_properties_section("draw", &PathDebugNode::properties_path_choice);
    if (ImGui::CollapsingHeader("Filter", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_properties_section("filter", &PathDebugNode::properties_filter);
    }
    if (ImGui::CollapsingHeader("Drawing", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_properties_section("drawing", &PathDebugNode::properties_line_style);
        ImGui::TextDisabled("vertices:");
        ImGui::SameLine();
        ImGui::TextColored(to_imvec4(path_debug_unpack_rgb(PATH_DEBUG_START_COLOR)), "start");
        ImGui::SameLine();
        ImGui::TextColored(to_imvec4(path_debug_unpack_rgb(PATH_DEBUG_END_COLOR)), "end");
        ImGui::SameLine();
        ImGui::TextColored(to_imvec4(path_debug_unpack_rgb(PATH_DEBUG_ESCAPED_COLOR)), "escaped");
    }
    if (ImGui::CollapsingHeader("Selection")) {
        draw_properties_section("selection", &PathDebugNode::properties_selection_details);
    }
    if (ImGui::CollapsingHeader("Drawn paths")) {
        draw_drawn_paths();
    }
}

void PathDebugNode::draw_view_tab() {
    draw_properties_section("view", &PathDebugNode::properties_view);
}

void PathDebugNode::draw_analysis_tab() {
    if (ImGui::CollapsingHeader("Sampling")) {
        path_statistics_shown = true;
        draw_sampling_section();
    }
    if (ImGui::CollapsingHeader("BSDF check")) {
        path_statistics_shown = true;
        draw_bsdf_section();
    }
    if (ImGui::CollapsingHeader("Pixel")) {
        draw_pixel_section();
    }
    if (ImGui::CollapsingHeader("Histograms")) {
        path_statistics_shown = true;
        draw_histograms_section();
    }
}

void PathDebugNode::draw_pixel_section() {
    pixel_section_shown = true;
    ImGui::TextDisabled("sample statistics, no reference needed");
    ImGui::SeparatorText(
        selection_visible()
            ? fmt::format("pixel {}, {}", selected_pixel.x, selected_pixel.y).c_str()
        : has_selection ? "pixel · off screen"
                        : "image centre");
    if (!selected_offscreen && !pixel_history.empty()) {
        const PixelSample& last = pixel_history.back();
        ImGui::Text("mean %.4g · variance %.4g · relative standard error %.3f · %u paths",
                    last.mean, last.variance, last.rel_error, latest_readback.probe.selected_count);
        const auto plot = [&](const char* title, float PixelSample::* const value,
                              const float min) {
            std::vector<float> values;
            values.reserve(pixel_history.size());
            for (const PixelSample& sample : pixel_history) {
                values.push_back(sample.*value);
            }
            ImGui::TextUnformatted(title);
            ImGui::PlotLines(fmt::format("##{}", title).c_str(), values.data(),
                             static_cast<int>(values.size()), 0, nullptr, min, FLT_MAX,
                             ImVec2(-FLT_MIN, 50));
        };
        plot("mean", &PixelSample::mean, FLT_MAX);
        plot("variance", &PixelSample::variance, 0.f);
        plot("relative standard error", &PixelSample::rel_error, 0.f);
    }
    if (ImGui::Button("Reset")) {
        pixel_stats_dirty = true;
    }
}

void PathDebugNode::draw_histograms_section() const {
    const auto& stats = latest_readback.stats;
    const auto histogram = [](const char* title, const float* values, const uint32_t bins) {
        ImGui::TextUnformatted(title);
        ImGui::PlotHistogram(fmt::format("##{}", title).c_str(), values, static_cast<int>(bins), 0,
                             nullptr, 0.f, FLT_MAX, ImVec2(-FLT_MIN, 60));
    };
    const auto counts = [&](const uint32_t offset, const uint32_t bins) {
        std::array<float, PATH_DEBUG_HIST_SMALL_BINS> values{};
        for (uint32_t i = 0; i < bins; i++) {
            values[i] = static_cast<float>(stats[offset + i]);
        }
        return values;
    };
    const auto count_histogram = [&](const char* title, const uint32_t offset,
                                     const uint32_t bins) {
        histogram(title, counts(offset, bins).data(), bins);
    };
    count_histogram("path luminance (log2)", PATH_DEBUG_STATS_LUM, PATH_DEBUG_HIST_SMALL_BINS);
    count_histogram("pdf (log2)", PATH_DEBUG_STATS_PDF, PATH_DEBUG_HIST_SMALL_BINS);
    count_histogram("paths by scatter events", PATH_DEBUG_STATS_SCATTER, PATH_DEBUG_SCATTER_BINS);

    std::array<float, PATH_DEBUG_SCATTER_BINS> contrib{};
    std::array<float, PATH_DEBUG_SCATTER_BINS> mean{};
    float total = 0.f;
    for (uint32_t i = 0; i < PATH_DEBUG_SCATTER_BINS; i++) {
        contrib[i] = stat_float(stats, PATH_DEBUG_STATS_SCATTER_CONTRIB + i);
        const uint32_t n = stats[PATH_DEBUG_STATS_SCATTER + i];
        mean[i] = n > 0 ? contrib[i] / static_cast<float>(n) : 0.f;
        total += contrib[i];
    }
    histogram("contribution by scatter events", contrib.data(), PATH_DEBUG_SCATTER_BINS);
    histogram("mean contribution by scatter events", mean.data(), PATH_DEBUG_SCATTER_BINS);
    if (total > 0.f) {
        std::string shares;
        for (uint32_t i = 0; i < PATH_DEBUG_SCATTER_BINS; i++) {
            if (contrib[i] > 0.02f * total) {
                shares += fmt::format("{}: {:.0f}%  ", i, 100.f * contrib[i] / total);
            }
        }
        wrapped_text(fmt::format("contribution share by scatter events: {}", shares));
    }
}

void PathDebugNode::draw_bsdf_section() {
    const auto& stats = latest_readback.stats;
    if (!gbuffer_connected) {
        ImGui::TextDisabled("gbuffer required");
        return;
    }
    if (!selection_visible()) {
        ImGui::TextDisabled(
            "%s",
            has_selection
                ? "selection off screen"
                : fmt::format("selection required · {}+click a surface", modifier_name()).c_str());
        return;
    }

    // 1. the incident direction every check uses
    ImGui::SeparatorText("incident direction");
    bool wi_changed = false;
    if (direction_disc("wi", wi_theta_deg, wi_phi_deg)) {
        wi_mode = static_cast<int32_t>(PATH_DEBUG_WI_MANUAL);
        wi_changed = true;
    }
    ImGui::SetItemTooltip("Drag to set the incident direction.");
    ImGui::SameLine();
    ImGui::BeginGroup();
    wi_changed |= segmented("incident", wi_mode, WI_NAMES);
    if (wi_mode == static_cast<int32_t>(PATH_DEBUG_WI_MANUAL)) {
        ImGui::Text("polar %.1f° · azimuth %.1f°", wi_theta_deg, wi_phi_deg);
    }
    bool below = wi_theta_deg > 90.f;
    if (ImGui::Checkbox("from below", &below)) {
        wi_theta_deg = 180.f - wi_theta_deg;
        wi_changed = true;
    }
    ImGui::EndGroup();
    if (wi_changed) {
        maps_dirty = true;
    }

    // 2. the validation suite
    ImGui::SeparatorText("validation");
    if (ImGui::Button("check BSDF")) {
        check_requested = true;
    }
    ImGui::SameLine();
    const uint32_t check_draws_done = stats[PATH_DEBUG_CHECK_DRAWS];
    ImGui::TextDisabled(
        "%s", check_draws_done < PATH_DEBUG_CHECK_TARGET_DRAWS
                  ? fmt::format("{} draws, a check tops them up to {} + {}² quadrature",
                                si_count(check_draws_done), si_count(PATH_DEBUG_CHECK_TARGET_DRAWS),
                                PATH_DEBUG_CHECK_DIM)
                        .c_str()
                  : fmt::format("{} draws + {}² quadrature", si_count(check_draws_done),
                                PATH_DEBUG_CHECK_DIM)
                        .c_str());
    const std::vector<BSDFVerdict> verdicts = bsdf_verdicts(stats);
    if (verdicts.empty()) {
        return;
    }
    for (const BSDFVerdict& verdict : verdicts) {
        check_row(verdict);
    }

    const auto draws = static_cast<float>(check_draws_done);
    const float3 albedo_mc = check_albedo(stats);
    ImGui::Text("alpha %.3g · %.1f%% valid · %.1f%% transmitted · albedo (%.4f, %.4f, %.4f)",
                stat_float(stats, PATH_DEBUG_CHECK_ALPHA),
                100.f * static_cast<float>(stats[PATH_DEBUG_CHECK_VALID]) / draws,
                100.f * static_cast<float>(stats[PATH_DEBUG_CHECK_TRANSMITTED]) / draws,
                albedo_mc.x, albedo_mc.y, albedo_mc.z);

    // 3. total variation against the pdf
    ImGui::SeparatorText("total variation to pdf");
    if (stats[PATH_DEBUG_STATS_CHECK_SAMPLES] > 0) {
        ImGui::Text("BSDF sample() %.2f%%",
                    total_variation_percent(stats, PATH_DEBUG_STATS_TV_CHECK));
        ImGui::SameLine();
    }
    ImGui::Text("BSDF eval() %.2f%%", total_variation_percent(stats, PATH_DEBUG_STATS_TV_EVAL));
    if (bsdf_comparable()) {
        ImGui::SameLine();
        ImGui::Text("recorded %.2f%%", total_variation_percent(stats, PATH_DEBUG_STATS_TV_RECORDS));
    }
}

void PathDebugNode::draw_drawn_paths() {
    drawn_paths_shown = true;

    bool any = false;
    if (ImGui::BeginTable("paths", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY,
                          ImVec2(0.f, 10.f * ImGui::GetTextLineHeightWithSpacing()))) {
        ImGui::TableSetupColumn("path", ImGuiTableColumnFlags_WidthFixed, 74.f);
        ImGui::TableSetupColumn("pixel", ImGuiTableColumnFlags_WidthFixed, 84.f);
        ImGui::TableSetupColumn("luminance", ImGuiTableColumnFlags_WidthFixed, 74.f);
        ImGui::TableSetupColumn("expression", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t t = 0; t < PATH_DEBUG_TOP_COUNT; t++) {
            const TopPath& entry = latest_readback.top[t];
            if (entry.slot == PATH_DEBUG_NO_PATH) {
                break;
            }
            any = true;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool is_isolated = isolate_path == entry.slot;
            if (ImGui::Selectable(fmt::format("#{}##path{}", entry.slot, t).c_str(), is_isolated,
                                  ImGuiSelectableFlags_SpanAllColumns)) {
                isolate_path = is_isolated ? std::nullopt : std::optional<uint32_t>(entry.slot);
            }
            ImGui::TableNextColumn();
            const uint2 pixel = path_record_unpack_pixel(entry.pixel);
            ImGui::Text("%u, %u", pixel.x, pixel.y);
            ImGui::TableNextColumn();
            ImGui::Text("%.4g", std::bit_cast<float>(entry.luminance));
            ImGui::TableNextColumn();
            const std::string events = path_expression_of(
                entry.classes, path_debug_unpack_count(entry.sample_and_scatter));
            ImGui::TextUnformatted(events.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton(fmt::format("set filter##expr{}", t).c_str())) {
                set_expression(events);
                scatter_range = int2(0, SCATTER_UNLIMITED);
                method_mask = 0;
                material_id = -1;
            }
            ImGui::SetItemTooltip("Filter by %s; resets the constraints.", events.c_str());
        }
        ImGui::EndTable();
    }
    if (!any) {
        ImGui::TextDisabled("no paths drawn");
    }

    if (!isolate_path) {
        return;
    }
    const IsolatedPath& isolated = latest_readback.isolated;
    if (isolated.valid == 0) {
        ImGui::TextDisabled("path #%u is no longer kept", *isolate_path);
    } else {
        const uint2 pixel = path_record_unpack_pixel(isolated.pixel);
        ImGui::Text("path #%u · pixel %u, %u · sample %u · frame %u · luminance %.4g",
                    *isolate_path, pixel.x, pixel.y,
                    path_debug_unpack_sample(isolated.sample_and_count),
                    latest_readback.frame.records_frame, std::bit_cast<float>(isolated.luminance));
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("show all")) {
        isolate_path.reset();
        return;
    }
    if (isolated.valid == 0) {
        return;
    }

    const uint32_t vertices = path_debug_unpack_count(isolated.sample_and_count);
    if (ImGui::BeginTable("vertices", 8,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY,
                          ImVec2(0.f, 12.f * ImGui::GetTextLineHeightWithSpacing()))) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 24.f);
        ImGui::TableSetupColumn("event", ImGuiTableColumnFlags_WidthFixed, 44.f);
        ImGui::TableSetupColumn("pdf", ImGuiTableColumnFlags_WidthFixed, 76.f);
        ImGui::TableSetupColumn("throughput", ImGuiTableColumnFlags_WidthFixed, 82.f);
        ImGui::TableSetupColumn("contribution", ImGuiTableColumnFlags_WidthFixed, 86.f);
        ImGui::TableSetupColumn("technique", ImGuiTableColumnFlags_WidthFixed, 74.f);
        ImGui::TableSetupColumn("material", ImGuiTableColumnFlags_WidthFixed, 60.f);
        ImGui::TableSetupColumn("position", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t v = 0; v < vertices; v++) {
            vertex_row(v, isolated.vertices.data() + (std::size_t{v} * PATH_RECORD_VERTEX_UINTS));
        }
        ImGui::EndTable();
    }
}

void PathDebugNode::draw_polar_profile() const {
    const auto& stats = latest_readback.stats;
    std::array<float, PATH_DEBUG_THETA_BANDS> records_curve{};
    std::array<float, PATH_DEBUG_THETA_BANDS> pdf_curve{};
    for (uint32_t band = 0; band < PATH_DEBUG_THETA_BANDS; band++) {
        const uint32_t base = PATH_DEBUG_STATS_THETA + (PATH_DEBUG_THETA_UINTS * band);
        const float solid_angle = stat_float(stats, base + 2);
        if (solid_angle <= 0.f) {
            continue;
        }
        records_curve[band] = static_cast<float>(stats[base + 0]) / solid_angle;
        pdf_curve[band] = stat_float(stats, base + 1) / solid_angle;
    }
    ImGui::SeparatorText("polar profile: pdf and recorded density");
    ImDrawList* const dl = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float plot_w = ImGui::GetContentRegionAvail().x - 10.f;
    const float plot_h = 110.f;
    dl->AddRectFilled(origin, ImVec2(origin.x + plot_w, origin.y + plot_h),
                      ImGui::GetColorU32(ImGuiCol_FrameBg), 4.f);
    const auto draw_curve = [&](const std::array<float, PATH_DEBUG_THETA_BANDS>& curve,
                                const ImU32 color, const float thickness) {
        const float vmax = *std::ranges::max_element(curve);
        if (vmax <= 0.f) {
            return;
        }
        std::array<ImVec2, PATH_DEBUG_THETA_BANDS> points;
        for (uint32_t band = 0; band < PATH_DEBUG_THETA_BANDS; band++) {
            const float x =
                origin.x + ((static_cast<float>(band) + 0.5f) / PATH_DEBUG_THETA_BANDS) * plot_w;
            const float y = origin.y + plot_h - 6.f - (curve[band] / vmax * (plot_h - 12.f));
            points[band] = ImVec2(x, y);
        }
        dl->AddPolyline(points.data(), PATH_DEBUG_THETA_BANDS, color, ImDrawFlags_None, thickness);
    };
    draw_curve(pdf_curve, LABEL_COLOR, 4.f);
    draw_curve(records_curve, RECORDED_COLOR, 1.5f);
    ImGui::Dummy(ImVec2(plot_w, plot_h + 4.f));
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(LABEL_COLOR), "pdf");
    ImGui::SameLine();
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(RECORDED_COLOR), "recorded");
}

void PathDebugNode::draw_sampling_section() {
    if (!maps_at_point()) {
        ImGui::TextDisabled("point selection required");
    } else if (readback_valid) {
        draw_sampling_fit();
        draw_polar_profile();
    }
    draw_properties_section("sampling", &PathDebugNode::properties_sampling);
}

void PathDebugNode::draw_sampling_fit() const {
    const auto& stats = latest_readback.stats;
    ImGui::Text("%s samples · pdf integral %.3f", si_count(stats[PATH_DEBUG_STATS_SAMPLES]).c_str(),
                stat_float(stats, PATH_DEBUG_STATS_PDF_INTEGRAL));
    const bool recorded = map_pdf_ref == static_cast<int32_t>(PATH_DEBUG_PDF_REF_RECORDED);
    const uint32_t dof = stats[PATH_DEBUG_STATS_CHI2_DOF];
    const uint32_t seen = stats[PATH_DEBUG_STATS_SEEN];
    const uint32_t batches = stats[PATH_DEBUG_STATS_BATCHES];
    if (!recorded && !bsdf_comparable()) {
        ImGui::TextDisabled("chi2 requires incident direction: camera ray");
    } else if ((seen & PATH_DEBUG_SEEN_RESAMPLED) != 0) {
        ImGui::TextDisabled("chi2 n/a: resampled paths carry no per-vertex pdf");
    } else if (!recorded && (seen & PATH_DEBUG_SEEN_CONNECTION) != 0) {
        ImGui::TextDisabled("chi2 of NEE draws requires test against: recorded pdf");
    } else if ((seen & PATH_DEBUG_SEEN_REUSED) != 0 && batches < PATH_DEBUG_MIN_BATCHES) {
        ImGui::TextDisabled("chi2 waits for batch %u of %u", batches, PATH_DEBUG_MIN_BATCHES);
    } else if (dof > 0) {
        const float reduced = reduced_chi2(stats);
        ImGui::TextColored(chi2_color(reduced, dof),
                           "chi2/dof %.2f over %u bins, expected 1 ± %.2f", reduced, dof,
                           chi2_sigma(dof));
    } else {
        ImGui::TextDisabled("chi2: too few samples");
    }
    if ((seen & PATH_DEBUG_SEEN_REUSED) != 0 && batches > 1) {
        const float lag_square = stat_float(stats, PATH_DEBUG_STATS_LAG_SQUARE);
        const float correlation =
            lag_square > 0.f ? stat_float(stats, PATH_DEBUG_STATS_LAG_PRODUCT) / lag_square : 0.f;
        const std::string batch_text =
            fmt::format("{} batches of {} frames · lag-1 correlation {:.2f}", batches,
                        map_batch_frames, correlation);
        if (correlation > 0.2f) {
            ImGui::TextColored(WARN_COLOR, "%s: raise batch frames", batch_text.c_str());
        } else {
            ImGui::TextDisabled("%s", batch_text.c_str());
        }
    }
    if (!recorded && stats[PATH_DEBUG_STATS_SPIKE_BINS] > 0) {
        ImGui::TextDisabled("excluded: %.1f%% of samples in %u quasi-delta bins",
                            stat_float(stats, PATH_DEBUG_STATS_SPIKE_MASS) * 100.f,
                            stats[PATH_DEBUG_STATS_SPIKE_BINS]);
    }
    ImGui::Separator();
}

std::string PathDebugNode::binned_caption() const {
    const char* const draws = DRAWS_NAMES.at(map_draws);
    // a renderer that does not report the technique records none of a kind
    const auto& stats = latest_readback.stats;
    const char* const none =
        stats[PATH_DEBUG_STATS_SAMPLES] == 0 && stats[PATH_DEBUG_STATS_COUNT] > 0
            ? " · none recorded"
            : "";
    return params.map_frame == PATH_DEBUG_FRAME_LOCAL ? fmt::format("({}){}", draws, none)
                                                      : fmt::format("({}) · world{}", draws, none);
}

void PathDebugNode::draw_panel_labels() const {
    if (params.map_panel_mask == 0) {
        return;
    }
    ImDrawList* const dl = ImGui::GetBackgroundDrawList();
    const auto& stats = latest_readback.stats;
    // the layout is in canvas pixels
    const float to_ui = 1.f / ImGui::GetIO().DisplayFramebufferScale.x;
    const float panel = static_cast<float>(panel_px) * to_ui;
    const float x0 = static_cast<float>(panel_left()) * to_ui;
    const float close = static_cast<float>(label_px(PATH_DEBUG_PANEL_CLOSE)) * to_ui;
    const float line = ImGui::GetTextLineHeight();
    const float shared_max = stat_float(stats, PATH_DEBUG_STATS_MAX_SHARED);
    const float integrand_max = stat_float(stats, PATH_DEBUG_STATS_MAX_INTEGRAND);
    const uint32_t dof = stats[PATH_DEBUG_STATS_CHI2_DOF];
    const auto atlas_rows = static_cast<float>(PATH_DEBUG_PANEL_ATLAS_ROWS);
    const auto atlas_columns = static_cast<float>(PATH_DEBUG_PANEL_ATLAS_COLUMNS);

    uint32_t slot = 0;
    for (uint32_t channel = 0; channel < PATH_DEBUG_MAP_CHANNEL_COUNT; channel++) {
        if ((params.map_panel_mask & (1u << channel)) == 0) {
            continue;
        }
        const float y0 = static_cast<float>(panel_top(slot++)) * to_ui;
        const ImVec2 uv0(static_cast<float>(channel % PATH_DEBUG_PANEL_ATLAS_COLUMNS) /
                             atlas_columns,
                         static_cast<float>(channel / PATH_DEBUG_PANEL_ATLAS_COLUMNS) / atlas_rows);
        const ImVec2 uv1(uv0.x + (1.f / atlas_columns), uv0.y + (1.f / atlas_rows));
        dl->AddImage(ImTextureRef(panel_texture_id), ImVec2(x0, y0), ImVec2(x0 + panel, y0 + panel),
                     uv0, uv1);
        dl->AddRect(ImVec2(x0 - 1.f, y0 - 1.f), ImVec2(x0 + panel + 1.f, y0 + panel + 1.f),
                    IM_COL32(90, 90, 90, 255));
        const ImVec2 close_min(x0 + panel - close, y0);
        dl->AddRectFilled(close_min, ImVec2(x0 + panel, y0 + close), IM_COL32(38, 38, 38, 255));
        dl->AddLine(ImVec2(close_min.x + 4.f, y0 + 4.f), ImVec2(x0 + panel - 4.f, y0 + close - 4.f),
                    LABEL_COLOR, 1.5f);
        dl->AddLine(ImVec2(close_min.x + 4.f, y0 + close - 4.f), ImVec2(x0 + panel - 4.f, y0 + 4.f),
                    LABEL_COLOR, 1.5f);
        if (!colored(channel)) {
            imgui_colorbar(dl, ImVec2(x0, y0 + panel + 1.f), ImVec2(x0 + panel, y0 + panel + 4.f),
                           diverging(channel) ? imgui_colormap_diverging : imgui_colormap_turbo);
        }

        std::string value;
        ImU32 value_color = LABEL_COLOR;
        switch (channel) {
        case PATH_DEBUG_MAP_INTEGRAND_NORMALIZED:
        case PATH_DEBUG_MAP_DENSITY:
        case PATH_DEBUG_MAP_RECORDED_PDF:
        case PATH_DEBUG_MAP_PDF:
            value = fmt::format("max {:.2g}/sr", shared_max);
            break;
        case PATH_DEBUG_MAP_SECOND_MOMENT:
            value = fmt::format("max {:.2g}/sr", stat_float(stats, PATH_DEBUG_STATS_MAX_MOMENT));
            break;
        case PATH_DEBUG_MAP_RADIANCE_COLOR:
            value = fmt::format("max {:.2g}", stat_float(stats, PATH_DEBUG_STATS_MAX_RADIANCE));
            break;
        case PATH_DEBUG_MAP_INTEGRAND:
        case PATH_DEBUG_MAP_INTEGRAND_COLOR:
            value = fmt::format("max {:.2g}/sr", integrand_max);
            break;
        case PATH_DEBUG_MAP_MEAN:
            value = fmt::format("max {:.2g}", stat_float(stats, PATH_DEBUG_STATS_MAX_MEAN));
            break;
        case PATH_DEBUG_MAP_RATIO:
        case PATH_DEBUG_MAP_COVERAGE:
            value = "0.25x - 4x";
            break;
        case PATH_DEBUG_MAP_ZSCORE:
            if (dof > 0) {
                const float reduced = reduced_chi2(stats);
                value = fmt::format("chi2/dof {:.2f}", reduced);
                value_color = ImGui::ColorConvertFloat4ToU32(chi2_color(reduced, dof));
            }
            break;
        case PATH_DEBUG_MAP_CHECK:
            if (stats[PATH_DEBUG_STATS_CHECK_SAMPLES] > 0) {
                value = si_count(stats[PATH_DEBUG_STATS_CHECK_SAMPLES]);
            }
            break;
        case PATH_DEBUG_MAP_EVAL:
            value = fmt::format("albedo {:.3g}", stat_float(stats, PATH_DEBUG_STATS_EVAL_NORM));
            break;
        default:
            break;
        }
        std::string caption;
        if (channel == PATH_DEBUG_MAP_DENSITY || channel == PATH_DEBUG_MAP_RECORDED_PDF ||
            channel == PATH_DEBUG_MAP_COVERAGE) {
            caption = binned_caption();
        } else if (channel == PATH_DEBUG_MAP_RADIANCE_COLOR && !bsdf_comparable()) {
            caption = "camera ray required";
        } else if (channel == PATH_DEBUG_MAP_RATIO || channel == PATH_DEBUG_MAP_ZSCORE) {
            const bool recorded = map_pdf_ref == static_cast<int32_t>(PATH_DEBUG_PDF_REF_RECORDED);
            caption = !recorded && !bsdf_comparable() ? "camera ray required"
                      : recorded                      ? "vs recorded pdf"
                                                      : "vs BSDF pdf()";
        }

        // the value moves below a name that leaves no room for it
        const float text_y = y0 + panel + 6.f;
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        const float value_width = ImGui::CalcTextSize(value.c_str()).x;
        const bool value_beside_name =
            ImGui::CalcTextSize(MAP_CHANNEL_NAMES[channel]).x + gap + value_width <= panel;
        shadowed_text(dl, ImVec2(x0, text_y), MAP_CHANNEL_NAMES[channel]);
        shadowed_text(dl,
                      ImVec2(x0 + panel - value_width, text_y + (value_beside_name ? 0.f : line)),
                      value.c_str(), value_color);
        const float caption_width =
            value_beside_name || value.empty() ? panel : panel - value_width - gap;
        shadowed_text(dl, ImVec2(x0, text_y + line), elided(caption, caption_width).c_str(),
                      IM_COL32(170, 170, 170, 255));
    }

    if (panel_overflow > 0) {
        const float height = static_cast<float>(label_area.y) * to_ui;
        const float column = height + (static_cast<float>(panel_overflow) * to_ui);
        const float track_x = x0 + panel + 4.f;
        const float top = height * static_cast<float>(panel_scroll) * to_ui / column;
        dl->AddRectFilled(ImVec2(track_x, 0.f), ImVec2(track_x + 3.f, height),
                          IM_COL32(0, 0, 0, 90));
        dl->AddRectFilled(ImVec2(track_x, top),
                          ImVec2(track_x + 3.f, top + (height * height / column)),
                          IM_COL32(200, 200, 200, 200));
    }
}

void PathDebugNode::draw_probe_tooltip() const {
    const Probe& probe = latest_readback.probe;
    if (!probe_at) {
        return;
    }
    std::string text;
    const uint32_t channel = probe.kind == PATH_DEBUG_PROBE_PANEL    ? params.probe_channel
                             : probe.kind == PATH_DEBUG_PROBE_SPHERE ? params.sphere_channel
                                                                     : PATH_DEBUG_MAP_NONE;
    if (channel != PATH_DEBUG_MAP_NONE) {
        const uint32_t shown = channel == PATH_DEBUG_MAP_RATIO || channel == PATH_DEBUG_MAP_ZSCORE
                                   ? PATH_DEBUG_MAP_DENSITY
                                   : channel;
        text = fmt::format("{} {:.3g}{} · pdf {:.3g}\nmean {:.3g} · {} samples",
                           MAP_CHANNEL_NAMES[shown], std::bit_cast<float>(probe.values[0]),
                           per_steradian(shown) ? "/sr" : "", std::bit_cast<float>(probe.values[1]),
                           std::bit_cast<float>(probe.values[2]), probe.count);
    } else if (probe.kind == PATH_DEBUG_PROBE_PIXEL) {
        text = fmt::format("mean {:.4g} · rel. std. error {:.3f}\nvariance {:.3g} · {} paths",
                           std::bit_cast<float>(probe.values[0]),
                           std::bit_cast<float>(probe.values[1]),
                           std::bit_cast<float>(probe.values[2]), probe.count);
    }
    if (text.empty()) {
        return;
    }
    ImDrawList* const dl = ImGui::GetBackgroundDrawList();
    const float to_ui = 1.f / ImGui::GetIO().DisplayFramebufferScale.x;
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    const float tx = std::min((probe_at->x * to_ui) + 16.f,
                              (static_cast<float>(label_area.x) * to_ui) - size.x - 8.f);
    const float ty = std::min((probe_at->y * to_ui) + 16.f,
                              (static_cast<float>(label_area.y) * to_ui) - size.y - 6.f);
    dl->AddRectFilled(ImVec2(tx - 5.f, ty - 3.f), ImVec2(tx + size.x + 5.f, ty + size.y + 3.f),
                      IM_COL32(15, 15, 18, 215), 4.f);
    shadowed_text(dl, ImVec2(tx, ty), text.c_str());
}

} // namespace merian
