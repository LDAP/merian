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
#include <set>
#include <span>
#include <string>
#include <utility>
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

float stat_float_float(const Stats& stats, const uint32_t slot) {
    return static_cast<float>(static_cast<double>(stat_float(stats, slot)) +
                              static_cast<double>(stat_float(stats, slot + 1)));
}

float3 check_albedo(const Stats& stats) {
    const float3 weight(stat_float_float(stats, PATH_DEBUG_CHECK_WEIGHT_TOTAL),
                        stat_float_float(stats, PATH_DEBUG_CHECK_WEIGHT_TOTAL + 2),
                        stat_float_float(stats, PATH_DEBUG_CHECK_WEIGHT_TOTAL + 4));
    return weight / static_cast<float>(std::max(stats[PATH_DEBUG_CHECK_DRAWS], 1u));
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

bool is_zscore(const uint32_t channel) {
    return channel == PATH_DEBUG_MAP_ZSCORE_SCATTER || channel == PATH_DEBUG_MAP_ZSCORE_NEE ||
           channel == PATH_DEBUG_MAP_ZSCORE_CHECK;
}

bool diverging(const uint32_t channel) {
    return is_zscore(channel) || channel == PATH_DEBUG_MAP_COVERAGE;
}

std::pair<float, uint32_t> panel_chi2(const Stats& stats, const uint32_t channel) {
    switch (channel) {
    case PATH_DEBUG_MAP_ZSCORE_SCATTER:
    case PATH_DEBUG_MAP_ZSCORE_NEE: {
        const uint32_t set =
            channel == PATH_DEBUG_MAP_ZSCORE_NEE ? PATH_DEBUG_SET_NEE : PATH_DEBUG_SET_SCATTER;
        return {stat_float(stats, path_debug_set_stat(set, PATH_DEBUG_SET_CHI2)),
                stats[path_debug_set_stat(set, PATH_DEBUG_SET_DOF)]};
    }
    default:
        return {stat_float(stats, PATH_DEBUG_STATS_CHECK_CHI2), stats[PATH_DEBUG_STATS_CHECK_DOF]};
    }
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

float small_button_width(const char* label) {
    return ImGui::CalcTextSize(label, nullptr, true).x + (2.f * ImGui::GetStyle().FramePadding.x);
}

float line_end() {
    return ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
}

void same_line_if_fits(const float width, const float spacing = -1.f) {
    const float gap = spacing < 0.f ? ImGui::GetStyle().ItemSpacing.x : spacing;
    if (ImGui::GetItemRectMax().x + gap + width <= line_end()) {
        ImGui::SameLine(0.f, spacing);
    }
}

void same_line_right_aligned(const char* label) {
    const float width = small_button_width(label);
    const float gap = line_end() - width - ImGui::GetItemRectMax().x;
    if (gap >= ImGui::GetStyle().ItemSpacing.x) {
        ImGui::SameLine(0.f, gap);
    } else {
        ImGui::SetCursorScreenPos(
            ImVec2(std::max(ImGui::GetCursorScreenPos().x, line_end() - width),
                   ImGui::GetCursorScreenPos().y));
    }
}

void faint_separator() {
    ImVec4 color = ImGui::GetStyleColorVec4(ImGuiCol_Separator);
    color.w *= 0.35f;
    ImGui::PushStyleColor(ImGuiCol_Separator, color);
    ImGui::Separator();
    ImGui::PopStyleColor();
}

void wrapped_disabled(const std::string& text) {
    ImGui::PushTextWrapPos(0.f);
    ImGui::TextDisabled("%s", text.c_str());
    ImGui::PopTextWrapPos();
}

bool begin_label_table(const char* id, const float label_width) {
    const float width = std::min(label_width, 0.5f * ImGui::GetContentRegionAvail().x);
    if (!ImGui::BeginTable(id, 2)) {
        return false;
    }
    ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, width);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
    return true;
}

void label_cell(const char* label) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::PushTextWrapPos(0.f);
    ImGui::TextUnformatted(label);
    ImGui::PopTextWrapPos();
    ImGui::TableNextColumn();
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
    enum class Status { OK, FAIL, APPROXIMATE, NO_STATEMENT, VALUE };
    const char* label;
    Status status;
    std::string detail;
};

BSDFVerdict::Status passed(const bool ok) {
    return ok ? BSDFVerdict::Status::OK : BSDFVerdict::Status::FAIL;
}

void check_row(const BSDFVerdict& verdict) {
    label_cell(verdict.label);
    ImGui::PushTextWrapPos(0.f);
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
    case BSDFVerdict::Status::VALUE:
        ImGui::TextUnformatted(verdict.detail.c_str());
        break;
    }
    ImGui::PopTextWrapPos();
}

struct BSDFVerdicts {
    std::vector<BSDFVerdict> valid;
    std::vector<BSDFVerdict> sampling;
};

BSDFVerdicts bsdf_verdicts(const Stats& stats) {
    const uint32_t draws = stats[PATH_DEBUG_CHECK_DRAWS];
    if (draws == 0 || stats[PATH_DEBUG_CHECK_CELLS] == 0) {
        return {};
    }

    const float3 albedo_mc = check_albedo(stats);
    const float albedo_lum = yuv_luminance(albedo_mc);
    const float albedo_max = std::max({albedo_mc.x, albedo_mc.y, albedo_mc.z});
    const float albedo_stderr =
        std::sqrt(stat_float_float(stats, PATH_DEBUG_CHECK_WEIGHT_M2_TOTAL)) /
        static_cast<float>(draws);
    const uint32_t transmitted = stats[PATH_DEBUG_CHECK_TRANSMITTED];
    const uint32_t continuous =
        std::max(stats[PATH_DEBUG_CHECK_VALID] - stats[PATH_DEBUG_CHECK_SINGULAR], 1u);
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

    BSDFVerdicts verdicts;
    // 1. valid
    const float mc_tol = std::max(6.f * albedo_stderr, 1e-3f);
    const bool energy_ok = albedo_max <= 1.f + mc_tol;
    verdicts.valid.push_back(
        {"energy conservation", passed(energy_ok),
         energy_ok ? fmt::format("albedo {:.4f} ± {:.4f}", albedo_max, albedo_stderr)
                   : fmt::format("albedo {:.4f}{}", albedo_max,
                                 transmitted > 0 ? ", refraction scales by eta²" : "")});
    verdicts.valid.push_back(identity("reciprocity", PATH_DEBUG_CHECK_OFF_RECIPROCITY,
                                      std::max(stats[PATH_DEBUG_CHECK_RECIPROCITY_CELLS], 1u),
                                      stat_float(stats, PATH_DEBUG_CHECK_ERR_RECIPROCITY)));
    const bool finite_ok = bad == 0 && bad_cells == 0;
    verdicts.valid.push_back(
        {"finite and non-negative", passed(finite_ok),
         finite_ok ? fmt::format("{} samples", si_count(draws))
                   : fmt::format("{} samples, {} cells", si_count(bad), si_count(bad_cells))});
    verdicts.valid.push_back(identity("weight = BSDF·cos/pdf", PATH_DEBUG_CHECK_OFF_WEIGHT,
                                      continuous, stat_float(stats, PATH_DEBUG_CHECK_ERR_WEIGHT)));
    // an approximation for demodulation, without the eta^2 radiance scaling of refraction
    const float albedo_fn = yuv_luminance(stat_rgb(stats, PATH_DEBUG_CHECK_ALBEDO));
    const float albedo_rel = (albedo_fn - albedo_lum) / std::max({albedo_fn, albedo_lum, 1e-6f});
    const float refracted = stat_float_float(stats, PATH_DEBUG_CHECK_TRANSMITTED_WEIGHT_TOTAL) /
                            static_cast<float>(draws);
    const std::string albedo_detail =
        fmt::format("{:.4f} vs {:.4f}, {:+.0f}%", albedo_fn, albedo_lum, albedo_rel * 100.f);
    if (std::abs(albedo_rel) <= 0.15f) {
        verdicts.valid.push_back({"get_albedo()", BSDFVerdict::Status::OK, albedo_detail});
    } else if (refracted > 0.f) {
        verdicts.valid.push_back({"get_albedo()", BSDFVerdict::Status::NO_STATEMENT,
                                  fmt::format("{:.4f} vs {:.4f}, {:.4f} refracted carries eta²",
                                              albedo_fn, albedo_lum, refracted)});
    } else {
        verdicts.valid.push_back({"get_albedo()", BSDFVerdict::Status::APPROXIMATE, albedo_detail});
    }

    // 2. sample() draws pdf()
    const uint32_t dof = stats[PATH_DEBUG_STATS_CHECK_DOF];
    if (dof > 0) {
        const float reduced =
            stat_float(stats, PATH_DEBUG_STATS_CHECK_CHI2) / static_cast<float>(dof);
        const float deviation = std::abs(reduced - 1.f) / chi2_sigma(dof);
        verdicts.sampling.push_back(
            {"chi2 of the draws", passed(deviation < 5.f),
             fmt::format("chi2/dof {:.2f} over {} bins, expected 1 ± {:.2f}", reduced, dof,
                         chi2_sigma(dof))});
    }
    verdicts.sampling.push_back(identity("sampled pdf = pdf()", PATH_DEBUG_CHECK_OFF_PDF,
                                         continuous, stat_float(stats, PATH_DEBUG_CHECK_ERR_PDF)));
    const float pdf_integral = stat_float(stats, PATH_DEBUG_CHECK_PDF_INTEGRAL);
    const float continuous_share =
        static_cast<float>(stats[PATH_DEBUG_CHECK_VALID] - stats[PATH_DEBUG_CHECK_SINGULAR]) /
        static_cast<float>(draws);
    const float pdf_rel = std::abs(pdf_integral - continuous_share) /
                          std::max({pdf_integral, continuous_share, 1e-6f});
    verdicts.sampling.push_back(integral(
        "pdf integrates to 1", pdf_rel <= 1e-2f,
        fmt::format("∫pdf dω {:.4f} vs continuous draws {:.4f}", pdf_integral, continuous_share)));
    const float eval_lum = yuv_luminance(stat_rgb(stats, PATH_DEBUG_CHECK_EVAL_INTEGRAL));
    const float continuous_albedo =
        stat_float_float(stats, PATH_DEBUG_CHECK_CONTINUOUS_WEIGHT_TOTAL) /
        static_cast<float>(draws);
    const float eval_rel =
        std::abs(eval_lum - continuous_albedo) / std::max({eval_lum, continuous_albedo, 1e-6f});
    verdicts.sampling.push_back(
        integral("draws integrate BSDF·cos", eval_rel <= 0.02f,
                 fmt::format("∫BSDF·cos dω {:.4f} vs continuous draws {:.4f}", eval_lum,
                             continuous_albedo)));
    if (stats[PATH_DEBUG_STATS_CHECK_SAMPLES] > 0) {
        verdicts.sampling.push_back(
            {"total variation", BSDFVerdict::Status::VALUE,
             fmt::format("{:.2f} % between the draws and pdf()",
                         total_variation_percent(stats, PATH_DEBUG_STATS_TV_CHECK))});
    }

    return verdicts;
}

void vertex_row(const uint32_t index, const uint32_t* const words, const std::string& material) {
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
    ImGui::Text("%.4g",
                yuv_luminance(path_record_unpack_rgb(words[PATH_RECORD_VERTEX_THROUGHPUT])));
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
    if (!terminal) {
        ImGui::TextUnformatted(material.c_str());
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

bool segmented(const char* id,
               int32_t& value,
               const std::span<const char* const> names,
               const std::span<const int32_t> shown = {}) {
    ImGui::PushID(id);
    bool changed = false;
    bool first = true;
    for (std::size_t i = 0; i < names.size(); i++) {
        if (!shown.empty() && std::ranges::find(shown, static_cast<int32_t>(i)) == shown.end()) {
            continue;
        }
        if (!first) {
            same_line_if_fits(small_button_width(names[i]), 1.f);
        }
        first = false;
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
            ImGui::PushItemWidth(-std::min(10.f * font, 0.4f * ImGui::GetContentRegionAvail().x));
            (this->*content)();
            ImGui::PopItemWidth();
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    };
    if (ImGui::BeginTabBar("tabs")) {
        tab("Paths", &PathDebugNode::draw_paths_tab);
        tab("Filter", &PathDebugNode::draw_filter_tab);
        tab("View", &PathDebugNode::draw_view_tab);
        tab("Analysis", &PathDebugNode::draw_analysis_tab);
        tab("Preferences", &PathDebugNode::draw_preferences_tab);
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
        const float font = ImGui::GetFontSize();
        labeled_combo("image", backdrop, BACKDROP_NAMES, 15.f * font);
        ImGui::SameLine();
        ImGui::TextUnformatted("exposure");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(5.f * font);
        ImGui::DragFloat("##exposure", &image_exposure, 0.01f, 0.f, FLT_MAX, "%.2f");
        if (path_debug_backdrop_accumulated(static_cast<uint32_t>(backdrop))) {
            ImGui::SameLine();
            if (ImGui::Button("reset")) {
                image_dirty = true;
            }
        }
        if (backdrop_missing) {
            ImGui::SameLine();
            ImGui::TextColored(WARN_COLOR, "%s required", missing_image_input());
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

    // 1. capture state, path counts and the export
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
            "probability %.3g\n%s drawn paths match%s",
            si_count(latest_readback.frame.paths).c_str(), si_count(record_path_capacity).c_str(),
            params.keep_prob, si_count(latest_readback.frame.matched).c_str(),
            capture == CAPTURE_FROZEN
                ? fmt::format("\nheld from frame {}", latest_readback.frame.records_frame).c_str()
                : "");
    }
    same_line_right_aligned("Export");
    if (ImGui::SmallButton("Export")) {
        ImGui::OpenPopup("Export");
    }
    ImGui::PushItemWidth(-8.f * font);
    if (ImGui::BeginPopup("Export")) {
        draw_properties_section("export", &PathDebugNode::properties_export);
        ImGui::EndPopup();
    }
    ImGui::PopItemWidth();

    // 2. what the selection is: a point, or a region's target
    draw_selection_choice();

    // 3. its details
    if (ImGui::TreeNode("selection")) {
        draw_properties_section("selection", &PathDebugNode::properties_selection_details);
        ImGui::TreePop();
    }
    ImGui::Separator();
}

void PathDebugNode::draw_selection_choice() {
    constexpr std::array<int32_t, 3> GEOMETRY = {0, TARGET_SPHERE + 1, TARGET_TRIANGLE + 1};
    constexpr std::array<int32_t, 4> SCENE_IDS = {TARGET_INSTANCE + 1, TARGET_NODE + 1,
                                                  TARGET_MESH + 1, TARGET_MATERIAL + 1};
    int32_t choice = selection_choice();
    bool chosen = segmented("geometry", choice, SELECTION_NAMES, GEOMETRY);
    same_line_right_aligned("Clear");
    ImGui::BeginDisabled(!has_selection);
    if (ImGui::SmallButton("Clear")) {
        clear_selection();
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("%s+Backspace", modifier_name());
    chosen |= segmented("scene ids", choice, SELECTION_NAMES, SCENE_IDS);
    if (chosen) {
        set_selection_choice(choice);
    }
    faint_separator();

    // a point's paths start at it
    int32_t match = select_kind == SELECT_POINT ? MATCH_STARTS : region_match;
    constexpr std::array<int32_t, 1> STARTS_ONLY = {MATCH_STARTS};
    if (segmented("match", match, MATCH_NAMES,
                  select_kind == SELECT_POINT ? std::span<const int32_t>(STARTS_ONLY)
                                              : std::span<const int32_t>()) &&
        select_kind == SELECT_REGION) {
        set_region_match(match);
    }
    const std::string status = selection_status();
    if (targets_ids() && region_target != TARGET_TRIANGLE) {
        same_line_if_fits(12.f * ImGui::GetFontSize());
        draw_target_choice(status);
        return;
    }
    if (status.empty()) {
        return;
    }
    same_line_if_fits(ImGui::CalcTextSize(status.c_str()).x);
    ImGui::PushTextWrapPos(0.f);
    if (has_selection && !targets_ids() && (selected_offscreen || point_hidden())) {
        ImGui::TextColored(WARN_COLOR, "%s", status.c_str());
    } else if (!has_selection || (targets_ids() ? !target_resolved() : !selected_pos_valid)) {
        ImGui::TextDisabled("%s", status.c_str());
    } else {
        ImGui::TextUnformatted(status.c_str());
    }
    ImGui::PopTextWrapPos();
}

std::string PathDebugNode::selection_status() const {
    if (!has_selection) {
        return fmt::format("no selection · {}+click the image", modifier_name());
    }
    if (targets_ids()) {
        return target_resolved() ? target_label() : "resolving surface";
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
    return "";
}

void PathDebugNode::draw_target_choice(const std::string& status) {
    ImGui::SetNextItemWidth(
        std::min(ImGui::GetContentRegionAvail().x, 24.f * ImGui::GetFontSize()));
    if (!ImGui::BeginCombo("##target", status.c_str(), ImGuiComboFlags_HeightLarge)) {
        return;
    }
    if (ImGui::Selectable("environment", target.environment)) {
        set_target({.environment = true});
    }
    const SceneHandle scene = current_scene.lock();
    const MaterialSystemHandle materials = material_system.lock();
    const auto material_of = [&](const Scene::MeshID mesh_id) {
        return static_cast<int32_t>(scene->get_mesh_infos()[mesh_id].mesh->material_id);
    };
    if (region_target == TARGET_MATERIAL && materials) {
        for (uint32_t id = 0; id < materials->get_material_count(); id++) {
            ImGui::PushID(static_cast<int>(id));
            if (ImGui::Selectable(material_label(id).c_str(),
                                  !target.environment &&
                                      target.material == static_cast<int32_t>(id))) {
                set_target({.material = static_cast<int32_t>(id)});
            }
            ImGui::PopID();
        }
    } else if (region_target == TARGET_MESH && scene) {
        for (Scene::MeshID id = 0; id < scene->get_mesh_infos().size(); id++) {
            if (!scene->get_mesh_infos()[id].mesh) {
                continue;
            }
            const auto mesh = static_cast<int32_t>(id);
            ImGui::PushID(mesh);
            if (ImGui::Selectable(mesh_label(mesh).c_str(),
                                  !target.environment && target.mesh == mesh)) {
                set_target({.mesh = mesh, .material = material_of(id)});
            }
            ImGui::PopID();
        }
    } else if (scene) {
        std::set<std::pair<Scene::NodeID, Scene::MeshID>> listed;
        for (const Scene::GeometryPlacement& placement : scene->get_geometry_placements()) {
            const Scene::MeshID listed_mesh =
                region_target == TARGET_NODE ? Scene::MeshID{0} : placement.mesh_id;
            if (!listed.emplace(placement.node_id, listed_mesh).second) {
                continue;
            }
            const auto mesh = static_cast<int32_t>(placement.mesh_id);
            const auto node = static_cast<int32_t>(placement.node_id);
            const std::string label =
                region_target == TARGET_NODE
                    ? node_label(node)
                    : fmt::format("{} @ {}", mesh_label(mesh), node_label(node));
            ImGui::PushID(static_cast<int>(listed.size()));
            const bool chosen = !target.environment && target.node == node &&
                                (region_target == TARGET_NODE || target.mesh == mesh);
            if (ImGui::Selectable(label.c_str(), chosen)) {
                set_target(region_target == TARGET_NODE
                               ? RegionTarget{.node = node}
                               : RegionTarget{.mesh = mesh,
                                              .node = node,
                                              .material = material_of(placement.mesh_id)});
            }
            ImGui::PopID();
        }
    }
    ImGui::EndCombo();
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
    case PATH_DEBUG_COLOR_SAMPLE_WEIGHT:
        bar(imgui_colormap_diverging,
            fmt::format("{:.3g}", std::exp2(-0.5f * PATH_DEBUG_SAMPLE_WEIGHT_STOPS)),
            fmt::format("{:.3g}", std::exp2(0.5f * PATH_DEBUG_SAMPLE_WEIGHT_STOPS)));
        break;
    case PATH_DEBUG_COLOR_SCATTER_EVENTS:
        bar(imgui_colormap_turbo, "0",
            fmt::format("{} scatter events", PATH_DEBUG_SCATTER_COLOR_EVENTS));
        break;
    case PATH_DEBUG_COLOR_TRANSPORTED:
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("the hue of the light along each segment");
        break;
    case PATH_DEBUG_COLOR_TECHNIQUE: {
        ImGui::AlignTextToFramePadding();
        bool first = true;
        for (const uint32_t method :
             {PATH_RECORD_METHOD_BSDF, PATH_RECORD_METHOD_GUIDING, PATH_RECORD_METHOD_LIGHT,
              PATH_RECORD_METHOD_REPLAYED_BSDF, PATH_RECORD_METHOD_REPLAYED_LIGHT,
              PATH_RECORD_METHOD_RECONNECTION, PATH_RECORD_METHOD_UNKNOWN}) {
            const char* const name =
                method == PATH_RECORD_METHOD_UNKNOWN ? "other" : PATH_RECORD_METHOD_NAMES[method];
            if (!std::exchange(first, false)) {
                same_line_if_fits(ImGui::CalcTextSize(name).x);
            }
            ImGui::TextColored(to_imvec4(path_debug_unpack_rgb(PATH_DEBUG_METHOD_COLORS[method])),
                               "%s", name);
        }
        break;
    }
    default:
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("one color per path");
        break;
    }
}

void PathDebugNode::draw_paths_tab() {
    draw_properties_section("color", &PathDebugNode::properties_color);
    if (overlay_color != static_cast<int32_t>(PATH_DEBUG_COLOR_OFF)) {
        draw_color_legend();
    }
    draw_properties_section("draw", &PathDebugNode::properties_path_choice);
    if (ImGui::CollapsingHeader("Line style", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_properties_section("line style", &PathDebugNode::properties_line_style);
        ImGui::TextDisabled("vertices:");
        ImGui::SameLine();
        ImGui::TextColored(to_imvec4(path_debug_unpack_rgb(PATH_DEBUG_START_COLOR)), "start");
        ImGui::SameLine();
        ImGui::TextColored(to_imvec4(path_debug_unpack_rgb(PATH_DEBUG_END_COLOR)), "end");
        ImGui::SameLine();
        ImGui::TextColored(to_imvec4(path_debug_unpack_rgb(PATH_DEBUG_ESCAPED_COLOR)), "escaped");
    }
    if (ImGui::CollapsingHeader("List")) {
        draw_list();
    }
}

void PathDebugNode::draw_filter_tab() {
    draw_properties_section("filter", &PathDebugNode::properties_filter);
    ImGui::SeparatorText("technique");
    const int columns = std::clamp(
        static_cast<int>(ImGui::GetContentRegionAvail().x / (9.f * ImGui::GetFontSize())), 1, 3);
    if (ImGui::BeginTable("techniques", columns, ImGuiTableFlags_SizingStretchSame)) {
        for (uint32_t m = 0; m < PATH_RECORD_METHOD_NAMES.size(); m++) {
            ImGui::TableNextColumn();
            bool set = (method_mask & (1u << m)) != 0;
            if (ImGui::Checkbox(PATH_RECORD_METHOD_NAMES[m], &set)) {
                set_method(m, set);
            }
        }
        ImGui::EndTable();
    }
    draw_properties_section("technique match", &PathDebugNode::properties_technique_match);
}

void PathDebugNode::draw_view_tab() {
    if (ImGui::CollapsingHeader("Overlay", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (segmented("type", false_color, FALSE_COLOR_NAMES)) {
            heat_dirty = true;
            pixel_stats_dirty = true;
        }
        draw_properties_section("overlay", &PathDebugNode::properties_false_color);
    }
    if (ImGui::CollapsingHeader("Sphere", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_properties_section("sphere", &PathDebugNode::properties_sphere);
    }
    if (ImGui::CollapsingHeader("Reference", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_properties_section("reference", &PathDebugNode::properties_reference);
    }
}

void PathDebugNode::draw_preferences_tab() {
    draw_properties_section("preferences", &PathDebugNode::properties_keys);
}

void PathDebugNode::draw_analysis_tab() {
    if (ImGui::CollapsingHeader("BSDF check", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_bsdf_section();
    }
    if (ImGui::CollapsingHeader("Panels", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_panel_choice();
    }
    if (ImGui::CollapsingHeader("Sampling")) {
        path_statistics_shown = true;
        draw_sampling_section();
    }
    if (ImGui::CollapsingHeader("Statistics")) {
        draw_statistics_section();
    }
}

void PathDebugNode::draw_panel_choice() {
    const float font = ImGui::GetFontSize();
    const auto drag_width = [font](const char* label) {
        return (5.f * font) + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize(label).x;
    };
    ImGui::Checkbox("show", &panels_enabled);
    ImGui::SetItemTooltip("While a point is selected; scroll over them to page.");
    same_line_if_fits(drag_width("size"));
    ImGui::SetNextItemWidth(5.f * font);
    ImGui::DragInt("size", &panel_size, 1.f, 64, 1024);
    same_line_if_fits(drag_width("exposure"));
    ImGui::SetNextItemWidth(5.f * font);
    ImGui::DragFloat("exposure", &map_exposure, 0.01f, 0.f, FLT_MAX, "%.2f");

    // per column a sampler: what it should follow, what it drew, the pdf it claims and the test
    const auto toggle = [this](const uint32_t channel) {
        ImGui::TableNextColumn();
        ImGui::Checkbox(MAP_CHANNEL_NAMES[channel], &panel_enabled[channel]);
        ImGui::SetItemTooltip("%s", MAP_CHANNEL_DESCRIPTIONS[channel]);
    };
    constexpr uint32_t SAMPLER_ROWS = 4;
    if (ImGui::BeginTable("panels", 3, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupColumn("continuations");
        ImGui::TableSetupColumn("NEE");
        ImGui::TableSetupColumn("BSDF");
        ImGui::TableHeadersRow();
        for (uint32_t channel = 0; channel < SAMPLER_ROWS * 3; channel++) {
            toggle(channel);
        }
        ImGui::EndTable();
    }
    if (ImGui::BeginTable("more panels", 3, ImGuiTableFlags_SizingStretchSame)) {
        for (uint32_t channel = SAMPLER_ROWS * 3; channel < PATH_DEBUG_MAP_CHANNEL_COUNT;
             channel++) {
            toggle(channel);
        }
        ImGui::EndTable();
    }
    if (ImGui::TreeNode("bins")) {
        draw_properties_section("bins", &PathDebugNode::properties_bins);
        ImGui::TreePop();
    }
}

void PathDebugNode::draw_statistics_section() {
    segmented("kind", statistics_kind, STATISTICS_NAMES);
    if (statistics_kind == STATISTICS_PIXEL) {
        draw_pixel_statistics();
    } else {
        path_statistics_shown = true;
        draw_light_statistics();
    }
}

void PathDebugNode::draw_pixel_statistics() {
    pixel_section_shown = true;
    wrapped_disabled(fmt::format(
        "{}, over every recorded path, no reference needed{}",
        selection_visible() ? fmt::format("pixel {}, {}", selected_pixel.x, selected_pixel.y)
        : has_selection     ? "pixel off screen"
                            : "image centre",
        false_color_filtered ? " · filtered" : ""));
    if (!selected_offscreen && !pixel_history.empty()) {
        const PixelSample& last = pixel_history.back();
        if (ImGui::BeginTable("pixel", 2, ImGuiTableFlags_SizingFixedFit)) {
            const auto row = [](const char* label, const char* formula, const std::string& value) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(label);
                ImGui::SetItemTooltip("%s", formula);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(value.c_str());
            };
            row("mean luminance", "mean(X), X what one recorded path adds to the pixel",
                fmt::format("{:.4g}", last.mean));
            row("sample count", "n, the recorded paths", fmt::format("{}", last.count));
            row("variance", "Var(X) = mean(X²) - mean(X)²", fmt::format("{:.4g}", last.variance));
            row("relative standard error", "sqrt(Var(X) / n) / mean(X)",
                fmt::format("{:.4f}", last.rel_error));
            row("max / mean", "max(X) / mean(X): fireflies lie far above 1",
                fmt::format("{:.3g}", last.max_ratio));
            row("zero fraction", "(paths with X = 0) / n",
                fmt::format("{:.1f} %", 100.f * last.zero_fraction));
            ImGui::EndTable();
        }
        std::vector<float> values;
        values.reserve(pixel_history.size());
        for (const PixelSample& sample : pixel_history) {
            values.push_back(sample.rel_error);
        }
        ImGui::TextUnformatted("relative standard error over the frames");
        ImGui::PlotLines("##rel error", values.data(), static_cast<int>(values.size()), 0, nullptr,
                         0.f, FLT_MAX, ImVec2(-FLT_MIN, 50));
    }
    if (ImGui::Button("reset")) {
        pixel_stats_dirty = true;
    }
}

void PathDebugNode::draw_light_statistics() const {
    const auto& stats = latest_readback.stats;
    const auto share_rows = [](const char* id,
                               const std::vector<std::pair<std::string, float>>& rows) {
        float total = 0.f;
        for (const auto& [label, value] : rows) {
            total += value;
        }
        if (!(total > 0.f)) {
            ImGui::TextDisabled("no light");
            return;
        }
        if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) {
            return;
        }
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("share", ImGuiTableColumnFlags_WidthStretch);
        for (const auto& [label, value] : rows) {
            const float share = value / total;
            if (share < 0.005f) {
                continue;
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(label.c_str());
            ImGui::TableNextColumn();
            ImGui::ProgressBar(share, ImVec2(-FLT_MIN, 0.f),
                               fmt::format("{:.1f} %", 100.f * share).c_str());
        }
        ImGui::EndTable();
    };

    ImGui::SeparatorText("light by scatter events");
    std::vector<std::pair<std::string, float>> by_scatter;
    for (uint32_t bin = 0; bin < PATH_DEBUG_SCATTER_BINS; bin++) {
        by_scatter.emplace_back(bin + 1 < PATH_DEBUG_SCATTER_BINS ? fmt::format("{}", bin)
                                                                  : fmt::format("{}+", bin),
                                stat_float(stats, PATH_DEBUG_STATS_SCATTER_CONTRIB + bin));
    }
    share_rows("scatter", by_scatter);

    ImGui::SeparatorText("light by the technique that reached it");
    std::vector<std::pair<std::string, float>> by_technique;
    for (uint32_t method = 0; method < PATH_RECORD_METHOD_COUNT; method++) {
        by_technique.emplace_back(method == PATH_RECORD_METHOD_UNKNOWN
                                      ? "camera ray, unknown"
                                      : PATH_RECORD_METHOD_NAMES[method],
                                  stat_float(stats, PATH_DEBUG_STATS_TECHNIQUE_CONTRIB + method));
    }
    share_rows("technique", by_technique);

    ImGui::SeparatorText("light paths by luminance");
    std::array<float, PATH_DEBUG_HIST_SMALL_BINS> counts{};
    for (uint32_t bin = 0; bin < PATH_DEBUG_HIST_SMALL_BINS; bin++) {
        counts[bin] = static_cast<float>(stats[PATH_DEBUG_STATS_LUM + bin]);
    }
    ImGui::PlotHistogram("##luminance", counts.data(), static_cast<int>(counts.size()), 0, nullptr,
                         0.f, FLT_MAX, ImVec2(-FLT_MIN, 60));
    const std::string low = fmt::format("2^{:.0f}", PATH_DEBUG_HIST_MIN_LOG);
    const std::string high = fmt::format("2^{:.0f}", PATH_DEBUG_HIST_MAX_LOG);
    ImGui::TextDisabled("%s", low.c_str());
    ImGui::SameLine(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x -
                                                         ImGui::CalcTextSize(high.c_str()).x));
    ImGui::TextDisabled("%s", high.c_str());
    ImGui::SetItemTooltip("A path's luminance on a log2 axis; the first bin holds the dark ones.");
}

void PathDebugNode::draw_list() {
    list_shown = true;
    const float font = ImGui::GetFontSize();
    constexpr ImGuiTableFlags TABLE_FLAGS = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                                            ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;

    bool any = false;
    if (ImGui::BeginTable("paths", 4, TABLE_FLAGS,
                          ImVec2(0.f, 10.f * ImGui::GetTextLineHeightWithSpacing()),
                          std::max(ImGui::GetContentRegionAvail().x, 30.f * font))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("path", ImGuiTableColumnFlags_WidthFixed, 5.5f * font);
        ImGui::TableSetupColumn("pixel", ImGuiTableColumnFlags_WidthFixed, 6.5f * font);
        ImGui::TableSetupColumn("luminance", ImGuiTableColumnFlags_WidthFixed, 5.5f * font);
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
    ImGui::PushTextWrapPos(0.f);
    if (isolated.valid == 0) {
        ImGui::TextDisabled("path #%u is no longer kept", *isolate_path);
    } else {
        const uint2 pixel = path_record_unpack_pixel(isolated.pixel);
        ImGui::Text("path #%u · pixel %u, %u · sample %u · frame %u · luminance %.4g",
                    *isolate_path, pixel.x, pixel.y,
                    path_debug_unpack_sample(isolated.sample_and_count),
                    latest_readback.frame.records_frame, std::bit_cast<float>(isolated.luminance));
    }
    ImGui::PopTextWrapPos();
    same_line_if_fits(small_button_width("show all"));
    if (ImGui::SmallButton("show all")) {
        isolate_path.reset();
        return;
    }
    if (isolated.valid == 0) {
        return;
    }

    const uint32_t vertices = path_debug_unpack_count(isolated.sample_and_count);
    if (ImGui::BeginTable("vertices", 8, TABLE_FLAGS,
                          ImVec2(0.f, 12.f * ImGui::GetTextLineHeightWithSpacing()),
                          std::max(ImGui::GetContentRegionAvail().x, 52.f * font))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 2.f * font);
        ImGui::TableSetupColumn("event", ImGuiTableColumnFlags_WidthFixed, 3.5f * font);
        ImGui::TableSetupColumn("pdf", ImGuiTableColumnFlags_WidthFixed, 6.f * font);
        ImGui::TableSetupColumn("throughput", ImGuiTableColumnFlags_WidthFixed, 6.5f * font);
        ImGui::TableSetupColumn("contribution", ImGuiTableColumnFlags_WidthFixed, 6.5f * font);
        ImGui::TableSetupColumn("technique", ImGuiTableColumnFlags_WidthFixed, 6.f * font);
        ImGui::TableSetupColumn("material", ImGuiTableColumnFlags_WidthFixed, 9.f * font);
        ImGui::TableSetupColumn("position", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t v = 0; v < vertices; v++) {
            const uint32_t* const words =
                isolated.vertices.data() + (std::size_t{v} * PATH_RECORD_VERTEX_UINTS);
            const uint32_t material =
                (words[PATH_RECORD_VERTEX_META] >> PATH_RECORD_MATERIAL_SHIFT) &
                PATH_RECORD_MATERIAL_MASK;
            vertex_row(v, words,
                       material != PATH_RECORD_MATERIAL_NONE ? material_label(material) : "");
        }
        ImGui::EndTable();
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
    path_statistics_shown = true;

    // 1. the incident direction every check uses
    ImGui::SeparatorText("incident direction");
    bool wi_changed = false;
    if (direction_disc("wi", wi_theta_deg, wi_phi_deg)) {
        wi_mode = static_cast<int32_t>(PATH_DEBUG_WI_MANUAL);
        wi_changed = true;
    }
    ImGui::SetItemTooltip("Drag to set the incident direction.");
    same_line_if_fits(ImGui::CalcTextSize("polar 180.0° · azimuth -180.0°").x);
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

    // 2. the material's verdicts
    if (ImGui::Button("check BSDF")) {
        check_requested = true;
    }
    const uint32_t check_draws_done = stats[PATH_DEBUG_CHECK_DRAWS];
    const std::string check_progress =
        check_topping_up
            ? fmt::format("{} of {} draws + {}² quadrature", si_count(check_draws_done),
                          si_count(PATH_DEBUG_CHECK_TARGET_DRAWS), PATH_DEBUG_CHECK_DIM)
        : check_draws_done < PATH_DEBUG_CHECK_TARGET_DRAWS
            ? fmt::format("{} draws, a check tops them up to {} + {}² quadrature",
                          si_count(check_draws_done), si_count(PATH_DEBUG_CHECK_TARGET_DRAWS),
                          PATH_DEBUG_CHECK_DIM)
            : fmt::format("{} draws + {}² quadrature", si_count(check_draws_done),
                          PATH_DEBUG_CHECK_DIM);
    same_line_if_fits(
        std::min(ImGui::CalcTextSize(check_progress.c_str()).x, 14.f * ImGui::GetFontSize()));
    wrapped_disabled(check_progress);
    const BSDFVerdicts verdicts = bsdf_verdicts(stats);
    const auto group = [](const char* title, const std::vector<BSDFVerdict>& rows) {
        if (rows.empty()) {
            return;
        }
        ImGui::SeparatorText(title);
        float label_width = 0.f;
        for (const BSDFVerdict& verdict : rows) {
            label_width = std::max(label_width, ImGui::CalcTextSize(verdict.label).x);
        }
        if (begin_label_table(title, label_width)) {
            for (const BSDFVerdict& verdict : rows) {
                check_row(verdict);
            }
            ImGui::EndTable();
        }
    };
    group("valid", verdicts.valid);
    group("sample() draws pdf()", verdicts.sampling);
    if (verdicts.valid.empty()) {
        return;
    }
    const auto draws = static_cast<float>(check_draws_done);
    const float3 albedo_mc = check_albedo(stats);
    wrapped_disabled(
        fmt::format("alpha {:.3g} · {:.1f}% valid · {:.1f}% transmitted · {:.1f}% "
                    "singular · albedo ({:.4f}, {:.4f}, {:.4f})",
                    stat_float(stats, PATH_DEBUG_CHECK_ALPHA),
                    100.f * static_cast<float>(stats[PATH_DEBUG_CHECK_VALID]) / draws,
                    100.f * static_cast<float>(stats[PATH_DEBUG_CHECK_TRANSMITTED]) / draws,
                    100.f * static_cast<float>(stats[PATH_DEBUG_CHECK_SINGULAR]) / draws,
                    albedo_mc.x, albedo_mc.y, albedo_mc.z));
}

void PathDebugNode::draw_sampling_section() {
    if (!maps_at_point()) {
        ImGui::TextDisabled("point selection required");
    } else if (readback_valid) {
        const auto& stats = latest_readback.stats;
        const uint32_t batches = stats[PATH_DEBUG_STATS_BATCHES];
        const auto test = [&](const char* question, const uint32_t set) {
            label_cell(question);
            const uint32_t seen = stats[path_debug_set_stat(set, PATH_DEBUG_SET_SEEN)];
            const uint32_t dof = stats[path_debug_set_stat(set, PATH_DEBUG_SET_DOF)];
            const bool reused = (seen & PATH_DEBUG_SEEN_REUSED) != 0;
            const uint32_t samples =
                stats[path_debug_set_stat(set, PATH_DEBUG_SET_SAMPLES)] +
                (set == PATH_DEBUG_SET_NEE
                     ? stats[path_debug_set_stat(PATH_DEBUG_SET_NEE_ENV, PATH_DEBUG_SET_SAMPLES)]
                     : 0u);
            if (samples == 0) {
                ImGui::TextDisabled("no draws");
            } else if ((seen & PATH_DEBUG_SEEN_RESAMPLED) != 0) {
                ImGui::TextDisabled("n/a: resampled draws record no pdf of their own");
            } else if (reused && batches < PATH_DEBUG_MIN_BATCHES) {
                ImGui::TextDisabled("waits for batch %u of %u", batches, PATH_DEBUG_MIN_BATCHES);
            } else if (dof > 0) {
                const float reduced =
                    stat_float(stats, path_debug_set_stat(set, PATH_DEBUG_SET_CHI2)) /
                    static_cast<float>(dof);
                ImGui::TextColored(chi2_color(reduced, dof),
                                   "chi2/dof %.2f over %u bins, expected 1 ± %.2f", reduced, dof,
                                   chi2_sigma(dof));
            } else {
                ImGui::TextDisabled("too few draws");
            }
            if (!reused || batches < 2) {
                return;
            }
            // reused draws are tested with batch means, which need uncorrelated batches
            const float lag_square =
                stat_float(stats, path_debug_set_stat(set, PATH_DEBUG_SET_LAG_SQUARE));
            const float correlation =
                lag_square > 0.f
                    ? stat_float(stats, path_debug_set_stat(set, PATH_DEBUG_SET_LAG_PRODUCT)) /
                          lag_square
                    : 0.f;
            if (correlation > 0.2f) {
                ImGui::TextColored(WARN_COLOR, "batches correlate %.2f: raise batch frames",
                                   correlation);
            }
        };
        constexpr std::array<const char*, 3> QUESTIONS = {
            "continuations follow their recorded pdfs", "NEE follows its recorded pdfs",
            "continuations fit the integrand"};
        float label_width = 0.f;
        for (const char* question : QUESTIONS) {
            label_width = std::max(label_width, ImGui::CalcTextSize(question).x);
        }
        if (begin_label_table("sampling", label_width)) {
            ImGui::PushTextWrapPos(0.f);
            test(QUESTIONS[0], PATH_DEBUG_SET_SCATTER);
            test(QUESTIONS[1], PATH_DEBUG_SET_NEE);
            label_cell(QUESTIONS[2]);
            if (stats[path_debug_set_stat(PATH_DEBUG_SET_SCATTER, PATH_DEBUG_SET_SAMPLES)] == 0 ||
                !(stat_float(stats, PATH_DEBUG_STATS_INTEGRAND) > 0.f)) {
                ImGui::TextDisabled("no draws or no light");
            } else {
                ImGui::Text("total variation %.1f %%",
                            total_variation_percent(stats, PATH_DEBUG_STATS_TV_INTEGRAND));
                ImGui::SetItemTooltip("0 %%: the draws follow BSDF·L·cos exactly, zero variance.");
            }
            ImGui::PopTextWrapPos();
            ImGui::EndTable();
        }
        wrapped_disabled(fmt::format(
            "{} continuations · {} NEE · {} guided draws",
            si_count(stats[path_debug_set_stat(PATH_DEBUG_SET_SCATTER, PATH_DEBUG_SET_SAMPLES)]),
            si_count(stats[path_debug_set_stat(PATH_DEBUG_SET_NEE, PATH_DEBUG_SET_SAMPLES)] +
                     stats[path_debug_set_stat(PATH_DEBUG_SET_NEE_ENV, PATH_DEBUG_SET_SAMPLES)]),
            si_count(stats[PATH_DEBUG_STATS_GUIDING_SAMPLES])));
    }
    draw_properties_section("sampling", &PathDebugNode::properties_sampling);
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
        if (channel != PATH_DEBUG_MAP_RADIANCE) {
            imgui_colorbar(dl, ImVec2(x0, y0 + panel + 1.f), ImVec2(x0 + panel, y0 + panel + 4.f),
                           diverging(channel) ? imgui_colormap_diverging : imgui_colormap_turbo);
        }

        std::string value;
        ImU32 value_color = LABEL_COLOR;
        switch (channel) {
        case PATH_DEBUG_MAP_SECOND_MOMENT:
            value = fmt::format("max {:.2g}/sr", stat_float(stats, PATH_DEBUG_STATS_MAX_MOMENT));
            break;
        case PATH_DEBUG_MAP_RADIANCE:
            value = fmt::format("max {:.2g}", stat_float(stats, PATH_DEBUG_STATS_MAX_RADIANCE));
            break;
        case PATH_DEBUG_MAP_SAMPLED_NEE:
        case PATH_DEBUG_MAP_RECORDED_NEE:
            value = fmt::format("max {:.2g}/sr", stat_float(stats, PATH_DEBUG_STATS_MAX_NEE));
            break;
        case PATH_DEBUG_MAP_COVERAGE:
            value = "0.25x - 4x";
            break;
        case PATH_DEBUG_MAP_ZSCORE_SCATTER:
        case PATH_DEBUG_MAP_ZSCORE_NEE:
        case PATH_DEBUG_MAP_ZSCORE_CHECK: {
            const auto [chi2, dof] = panel_chi2(stats, channel);
            if (dof > 0) {
                const float reduced = chi2 / static_cast<float>(dof);
                value = fmt::format("chi2/dof {:.2f}", reduced);
                value_color = ImGui::ColorConvertFloat4ToU32(chi2_color(reduced, dof));
            }
            break;
        }
        case PATH_DEBUG_MAP_CHECK:
            value = si_count(stats[PATH_DEBUG_STATS_CHECK_SAMPLES]);
            break;
        case PATH_DEBUG_MAP_EVAL:
            value = fmt::format("albedo {:.3g}", stat_float(stats, PATH_DEBUG_STATS_EVAL_NORM));
            break;
        default:
            value = fmt::format("max {:.2g}/sr", shared_max);
            break;
        }
        std::string caption;
        switch (channel) {
        case PATH_DEBUG_MAP_SAMPLED_SCATTER:
        case PATH_DEBUG_MAP_RECORDED_SCATTER:
            caption = "every continuation";
            break;
        case PATH_DEBUG_MAP_SAMPLED_NEE:
        case PATH_DEBUG_MAP_RECORDED_NEE:
            caption = "the light connections";
            break;
        case PATH_DEBUG_MAP_SAMPLED_GUIDING:
            caption = "the guided continuations";
            break;
        case PATH_DEBUG_MAP_ZSCORE_SCATTER:
            caption = "continuations vs recorded pdfs";
            break;
        case PATH_DEBUG_MAP_ZSCORE_NEE:
            caption = "NEE vs recorded pdfs";
            break;
        case PATH_DEBUG_MAP_ZSCORE_CHECK:
            caption = "sample() vs its pdfs";
            break;
        case PATH_DEBUG_MAP_RADIANCE:
            caption = bsdf_comparable() ? "" : "camera ray required";
            break;
        default:
            break;
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
        const float value = std::bit_cast<float>(probe.values[0]);
        const std::string shown = is_zscore(channel) ? fmt::format("z {:+.2f}", value)
                                  : channel == PATH_DEBUG_MAP_COVERAGE
                                      ? fmt::format("{:.3g}x", value)
                                      : fmt::format("{:.3g}", value);
        text = fmt::format("{} {}\nBSDF pdf() {:.3g} · {} draws · {:.3g} sr",
                           MAP_CHANNEL_NAMES[channel], shown, std::bit_cast<float>(probe.values[1]),
                           probe.count, std::bit_cast<float>(probe.values[2]));
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
