#include "merian-graph/nodes/path_debug/path_expression.hpp"

#include <algorithm>
#include <cctype>
#include <exception>

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
                    const uint32_t repeat_min,
                    const uint32_t repeat_max) {
    return scattering | (side << PATH_DEBUG_SIDE_SHIFT) | (repeat_min << PATH_DEBUG_MIN_SHIFT) |
           (repeat_max << PATH_DEBUG_MAX_SHIFT);
}

} // namespace

PathExpression compile_path_expression(const std::string& text) {
    const PathExpression invalid{.valid = false};
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

    const auto rest_is_blank = [&text](const std::size_t from) {
        return text.find_first_not_of(" \t", from) == std::string::npos;
    };

    while (i < text.size()) {
        const char c = text[i];
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            i++;
            continue;
        }
        const bool last = rest_is_blank(i + 1);
        const auto upper = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (upper == 'N' && last) {
            nee = true;
            i++;
            continue;
        }
        if ((c == 'C' && count == 0) || (upper == 'L' && last)) {
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
            const bool light = std::ranges::any_of(letters, [](const char letter) {
                return std::toupper(static_cast<unsigned char>(letter)) == 'L';
            });
            if (light && rest_is_blank(i)) {
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
            side = PATH_DEBUG_SIDE_ANY;
        }

        uint32_t repeat_min = 1;
        uint32_t repeat_max = 1;
        if (!parse_repeat(text, i, repeat_min, repeat_max)) {
            return invalid;
        }
        if (repeat_min > PATH_DEBUG_MIN_MASK || repeat_min > repeat_max ||
            repeat_max > PATH_DEBUG_REPEAT_INF) {
            return invalid;
        }

        tokens[count++] = pack_token(scattering, side, repeat_min, repeat_max);
    }

    if (nee && count == 0) {
        tokens[count++] =
            pack_token(PATH_DEBUG_CLS_ANY, PATH_DEBUG_SIDE_ANY, 0, PATH_DEBUG_REPEAT_INF);
    }

    PathExpression expression;
    for (uint32_t w = 0; w < expression.words.size(); w++) {
        expression.words[w] = tokens[2 * w] | (tokens[(2 * w) + 1] << PATH_DEBUG_TOKEN_BITS);
    }
    expression.token_count = count;
    expression.nee = nee;
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
