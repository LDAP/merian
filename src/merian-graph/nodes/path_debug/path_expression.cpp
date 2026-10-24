#include "merian-graph/nodes/path_debug/path_expression.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <string_view>

namespace merian {

namespace {

// A repeat after an event at i: * + ? {n} {n,m} {n,}; none leaves the bounds at 1.
bool parse_repeat(const std::string& text,
                  std::size_t& i,
                  uint32_t& repeat_min,
                  uint32_t& repeat_max) {
    if (i >= text.size()) {
        return true;
    }
    switch (text[i]) {
    case '*':
        repeat_min = 0;
        repeat_max = PATH_DEBUG_REPEAT_INF;
        i++;
        return true;
    case '+':
        repeat_max = PATH_DEBUG_REPEAT_INF;
        i++;
        return true;
    case '?':
        repeat_min = 0;
        i++;
        return true;
    case '{':
        break;
    default:
        return true;
    }
    const std::size_t end_pos = text.find('}', i);
    if (end_pos == std::string::npos) {
        return false;
    }
    const std::string range = text.substr(i + 1, end_pos - i - 1);
    const std::size_t comma = range.find(',');
    try {
        if (comma == std::string::npos) {
            repeat_min = repeat_max = std::stoul(range);
        } else {
            repeat_min = std::stoul(range.substr(0, comma));
            const std::string upper = range.substr(comma + 1);
            repeat_max = upper.empty() ? PATH_DEBUG_REPEAT_INF : std::stoul(upper);
        }
    } catch (const std::exception&) {
        return false;
    }
    i = end_pos + 1;
    return true;
}

uint32_t pack_token(const uint32_t scattering,
                    const uint32_t side,
                    const uint32_t labels,
                    const uint32_t repeat_min,
                    const uint32_t repeat_max) {
    return scattering | (side << PATH_DEBUG_SIDE_SHIFT) | (repeat_min << PATH_DEBUG_MIN_SHIFT) |
           (repeat_max << PATH_DEBUG_MAX_SHIFT) | (labels << PATH_DEBUG_LABEL_SHIFT);
}

// the techniques a quoted code names, 0 for an unknown code
uint32_t technique_labels(const std::string_view code) {
    static constexpr std::array<std::pair<std::string_view, uint32_t>, 8> CODES = {{
        {"b", 1u << PATH_RECORD_METHOD_BSDF},
        {"g", 1u << PATH_RECORD_METHOD_GUIDING},
        {"n", 1u << PATH_RECORD_METHOD_LIGHT},
        {"c", 1u << PATH_RECORD_METHOD_COSINE},
        {"u", 1u << PATH_RECORD_METHOD_UNIFORM},
        {"rrb", 1u << PATH_RECORD_METHOD_REPLAYED_BSDF},
        {"rrn", 1u << PATH_RECORD_METHOD_REPLAYED_LIGHT},
        {"rec", 1u << PATH_RECORD_METHOD_RECONNECTION},
    }};
    const auto it = std::ranges::find(CODES, code, &std::pair<std::string_view, uint32_t>::first);
    return it != CODES.end() ? it->second : 0u;
}

} // namespace

PathExpression compile_path_expression(const std::string& text) {
    const PathExpression invalid{.valid = false};
    PathExpression expression;
    expression.active = text.find_first_not_of(" \t") != std::string::npos;
    std::size_t i = 0;

    const auto masks = [](const std::string& event, uint32_t& scattering, uint32_t& side,
                          uint32_t& labels, bool& light) {
        for (std::size_t at = 0; at < event.size(); at++) {
            const char c = event[at];
            if (c == '\'') {
                const std::size_t close = event.find('\'', at + 1);
                const uint32_t technique =
                    close == std::string::npos
                        ? 0u
                        : technique_labels(std::string_view(event).substr(at + 1, close - at - 1));
                if (technique == 0) {
                    return false;
                }
                labels |= technique;
                at = close;
                continue;
            }
            switch (std::toupper(static_cast<unsigned char>(c))) {
            case 'S':
                scattering |= PATH_DEBUG_CLS_DELTA;
                break;
            case 'G':
                scattering |= PATH_DEBUG_CLS_GLOSSY;
                break;
            case 'D':
                scattering |= PATH_DEBUG_CLS_DIFFUSE;
                break;
            case 'U':
                scattering |= PATH_DEBUG_CLS_UNKNOWN;
                break;
            case 'R':
                side |= PATH_DEBUG_SIDE_REFLECT;
                break;
            case 'T':
                side |= PATH_DEBUG_SIDE_TRANSMIT;
                break;
            case 'L':
                light = true;
                break;
            case '.':
            case ' ':
                break;
            default:
                return false;
            }
        }
        return true;
    };

    while (i < text.size()) {
        const char c = text[i];
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            i++;
            continue;
        }
        if (c == 'C' && expression.token_count == 0) {
            i++;
            continue;
        }

        // one event: a bracket, a quoted technique or a letter
        std::size_t end_pos = i;
        if (c == '<' || c == '[') {
            end_pos = text.find(c == '<' ? '>' : ']', i);
        } else if (c == '\'') {
            end_pos = text.find('\'', i + 1);
        }
        if (end_pos == std::string::npos) {
            return invalid;
        }
        const bool bracket = c == '<' || c == '[';
        const std::string event =
            bracket ? text.substr(i + 1, end_pos - i - 1) : text.substr(i, end_pos - i + 1);
        i = end_pos + 1;

        uint32_t scattering = 0;
        uint32_t side = 0;
        uint32_t labels = 0;
        bool light = false;
        if (!masks(event, scattering, side, labels, light)) {
            return invalid;
        }
        // the light ends the expression and carries nothing of its own
        if (light) {
            if (text.find_first_not_of(" \t", i) != std::string::npos || scattering != 0 ||
                side != 0 || labels != 0) {
                return invalid;
            }
            break;
        }

        uint32_t repeat_min = 1;
        uint32_t repeat_max = 1;
        if (!parse_repeat(text, i, repeat_min, repeat_max)) {
            return invalid;
        }
        if (expression.token_count == PATH_DEBUG_MAX_TOKENS || repeat_min > PATH_DEBUG_MIN_MASK ||
            repeat_min > repeat_max || repeat_max > PATH_DEBUG_REPEAT_INF) {
            return invalid;
        }
        expression.tokens[expression.token_count++] =
            pack_token(scattering != 0 ? scattering : PATH_DEBUG_CLS_ANY,
                       side != 0 ? side : PATH_DEBUG_SIDE_ANY,
                       labels != 0 ? labels : PATH_DEBUG_LABEL_ANY, repeat_min, repeat_max);
    }
    return expression;
}

std::string path_event_name(const uint32_t lobe) {
    const uint32_t smoothness =
        lobe & (PATH_DEBUG_CLS_DELTA | PATH_DEBUG_CLS_GLOSSY | PATH_DEBUG_CLS_DIFFUSE);
    const char* const scattering = smoothness == PATH_DEBUG_CLS_DELTA     ? "S"
                                   : smoothness == PATH_DEBUG_CLS_GLOSSY  ? "G"
                                   : smoothness == PATH_DEBUG_CLS_DIFFUSE ? "D"
                                                                          : "U";
    return std::string((lobe & PATH_RECORD_LOBE_TRANSMISSION) != 0 ? "T" : "R") + scattering;
}

std::string path_expression_of(const uint32_t classes, const uint32_t events) {
    std::string expression = "C";
    const uint32_t shown = std::min(events, PATH_DEBUG_TOP_CLASSES);
    for (uint32_t e = 0; e < shown; e++) {
        expression +=
            "<" +
            path_event_name((classes >> (PATH_DEBUG_CLASS_BITS * e)) & PATH_RECORD_LOBE_MASK) + ">";
    }
    return shown < events ? expression + ".*" : expression + "<L.>";
}

} // namespace merian
