#include "merian-graph/nodes/path_debug/path_expression.hpp"

#include "merian-graph/nodes/path_debug/layout.slangh"

#include <algorithm>
#include <cctype>
#include <exception>

namespace merian {

PathExpression compile_path_expression(const std::string& text) {
    PathExpression invalid;
    invalid.valid = false;
    std::array<uint32_t, PATH_DEBUG_MAX_TOKENS> tokens{};
    uint32_t count = 0;
    bool nee = false;
    std::size_t i = 0;

    const auto letter_masks = [](const std::string& letters, uint32_t& scattering, uint32_t& side) {
        for (const char c : letters) {
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
            case '.':
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
        const bool last = text.find_first_not_of(" \t", i + 1) == std::string::npos;
        if (std::toupper(static_cast<unsigned char>(c)) == 'N' && last) {
            nee = true;
            i++;
            continue;
        }
        if ((c == 'C' && count == 0) ||
            (std::toupper(static_cast<unsigned char>(c)) == 'L' && last)) {
            i++;
            continue;
        }
        if (count == PATH_DEBUG_MAX_TOKENS) {
            return invalid;
        }

        std::string letters;
        if (c == '<' || c == '[') {
            const char closing = c == '<' ? '>' : ']';
            const std::size_t end_pos = text.find(closing, i);
            if (end_pos == std::string::npos) {
                return invalid;
            }
            letters = text.substr(i + 1, end_pos - i - 1);
            i = end_pos + 1;
            if ((letters.find('L') != std::string::npos ||
                 letters.find('l') != std::string::npos) &&
                text.find_first_not_of(" \t", i) == std::string::npos) {
                continue;
            }
        } else {
            letters = std::string(1, c);
            i++;
        }

        uint32_t scattering = 0;
        uint32_t side = 0;
        if (!letter_masks(letters, scattering, side)) {
            return invalid;
        }
        if (scattering == 0) {
            scattering = PATH_DEBUG_CLS_ANY;
        }
        if (side == 0) {
            side = PATH_DEBUG_SIDE_REFLECT | PATH_DEBUG_SIDE_TRANSMIT;
        }

        uint32_t repeat_min = 1;
        uint32_t repeat_max = 1;
        if (i < text.size()) {
            if (text[i] == '*') {
                repeat_min = 0;
                repeat_max = PATH_DEBUG_REPEAT_INF;
                i++;
            } else if (text[i] == '+') {
                repeat_min = 1;
                repeat_max = PATH_DEBUG_REPEAT_INF;
                i++;
            } else if (text[i] == '?') {
                repeat_min = 0;
                repeat_max = 1;
                i++;
            } else if (text[i] == '{') {
                const std::size_t end_pos = text.find('}', i);
                if (end_pos == std::string::npos) {
                    return invalid;
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
                    return invalid;
                }
                i = end_pos + 1;
            }
        }
        constexpr uint32_t min_limit = (1u << (PATH_DEBUG_MAX_SHIFT - PATH_DEBUG_MIN_SHIFT)) - 1;
        if (repeat_min > min_limit || repeat_min > repeat_max ||
            repeat_max > PATH_DEBUG_REPEAT_INF) {
            return invalid;
        }

        tokens[count++] = scattering | (side << PATH_DEBUG_SIDE_SHIFT) |
                          (repeat_min << PATH_DEBUG_MIN_SHIFT) |
                          (repeat_max << PATH_DEBUG_MAX_SHIFT);
    }

    if (nee && count == 0) {
        tokens[count++] =
            PATH_DEBUG_CLS_ANY |
            ((PATH_DEBUG_SIDE_REFLECT | PATH_DEBUG_SIDE_TRANSMIT) << PATH_DEBUG_SIDE_SHIFT) |
            (PATH_DEBUG_REPEAT_INF << PATH_DEBUG_MAX_SHIFT);
    }

    PathExpression expression;
    for (uint32_t w = 0; w < expression.words.size(); w++) {
        expression.words[w] = tokens[2 * w] | (tokens[(2 * w) + 1] << 16);
    }
    expression.token_count = count;
    expression.nee = nee;
    return expression;
}

std::string path_event_name(const uint32_t lobe) {
    const uint32_t smoothness =
        lobe & (PATH_DEBUG_CLS_DELTA | PATH_DEBUG_CLS_GLOSSY | PATH_DEBUG_CLS_DIFFUSE);
    const char* scattering = smoothness == PATH_DEBUG_CLS_DELTA     ? "S"
                             : smoothness == PATH_DEBUG_CLS_GLOSSY  ? "G"
                             : smoothness == PATH_DEBUG_CLS_DIFFUSE ? "D"
                                                                    : "U";
    return std::string((lobe & PATH_RECORD_LOBE_TRANSMISSION) != 0 ? "T" : "R") + scattering;
}

std::string path_expression_of(const uint32_t classes, const uint32_t events) {
    std::string expression = "C";
    const uint32_t shown = std::min(events, 8u);
    for (uint32_t e = 0; e < shown; e++) {
        expression += "<" + path_event_name((classes >> (4 * e)) & PATH_RECORD_LOBE_MASK) + ">";
    }
    return shown < events ? expression + ".*" : expression + "<L.>";
}

} // namespace merian
