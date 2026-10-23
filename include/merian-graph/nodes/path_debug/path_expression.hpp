#pragma once

#include "merian-graph/nodes/path_debug/layout.slangh"

#include <array>
#include <cstdint>
#include <string>

namespace merian {

struct PathExpression {
    std::array<uint32_t, PATH_DEBUG_MAX_TOKENS> tokens{};
    uint32_t token_count = 0;
    // blank text matches every light path
    bool active = false;
    bool valid = true;
};

// Compiles an expression after light path expressions (Heckbert 1990; OSL LPE) over the scatter
// events of a path from the camera to a light: S specular, G glossy, D diffuse, U unclassified, R
// reflected, T transmitted, '.' any; a quoted code for the technique that drew the event's
// outgoing direction, 'b' BSDF, 'g' guiding, 'n' NEE, 'c' cosine, 'u' uniform, 'rrb' replayed
// BSDF, 'rrn' replayed NEE, 'rec' reconnection, the last event's is how the light was reached;
// <D'g'> one event of all of it, [SG] of either; the repeats * + ? {n} {n,m}. The expression
// matches every event; C and a trailing L are implicit, L alone is a light the camera sees.
PathExpression compile_path_expression(const std::string& text);

std::string path_event_name(uint32_t lobe);

std::string path_expression_of(uint32_t classes, uint32_t events);

} // namespace merian
