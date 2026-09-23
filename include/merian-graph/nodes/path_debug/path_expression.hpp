#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace merian {

struct PathExpression {
    std::array<uint32_t, 4> words{};
    uint32_t token_count = 0;
    bool nee = false;
    bool valid = true;
};

// Compiles an expression after light path expressions (Heckbert 1990; OSL LPE): per scatter event
// S specular, G glossy, D diffuse, U unclassified, R reflected, T transmitted, '.' any, <RD>
// combined, [SG] alternatives, and the repeats * + ? {n} {n,m}. The expression matches a whole
// path; C and <L.> anchors are implicit. A trailing N instead matches the events up to a vertex
// that connected to a light by next-event estimation.
PathExpression compile_path_expression(const std::string& text);

std::string path_event_name(uint32_t lobe);

std::string path_expression_of(uint32_t classes, uint32_t events);

} // namespace merian
