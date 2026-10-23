#include <gtest/gtest.h>

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/path_expression.hpp"

using namespace merian;

namespace {

uint32_t token(const PathExpression& expression, const uint32_t index) {
    return expression.tokens[index];
}

uint32_t scattering(const uint32_t t) {
    return t & 0xFu;
}

uint32_t side(const uint32_t t) {
    return (t >> PATH_DEBUG_SIDE_SHIFT) & 0x3u;
}

uint32_t repeat_min(const uint32_t t) {
    return (t >> PATH_DEBUG_MIN_SHIFT) & 0x7u;
}

uint32_t repeat_max(const uint32_t t) {
    return (t >> PATH_DEBUG_MAX_SHIFT) & 0xFu;
}

uint32_t labels(const uint32_t t) {
    return t >> PATH_DEBUG_LABEL_SHIFT;
}

} // namespace

TEST(PathExpression, EmptyAcceptsEverything) {
    const PathExpression expression = compile_path_expression(" ");
    EXPECT_TRUE(expression.valid);
    EXPECT_FALSE(expression.active);
    EXPECT_EQ(expression.token_count, 0u);
}

TEST(PathExpression, LightAloneIsSeenByTheCamera) {
    const PathExpression expression = compile_path_expression("L");
    ASSERT_TRUE(expression.valid);
    EXPECT_TRUE(expression.active);
    EXPECT_EQ(expression.token_count, 0u);
}

TEST(PathExpression, AnchorsAreImplicit) {
    const PathExpression expression = compile_path_expression("C<RD><L.>");
    ASSERT_TRUE(expression.valid);
    ASSERT_EQ(expression.token_count, 1u);
    EXPECT_EQ(scattering(token(expression, 0)), PATH_DEBUG_CLS_DIFFUSE);
    EXPECT_EQ(side(token(expression, 0)), PATH_DEBUG_SIDE_REFLECT);
    EXPECT_EQ(repeat_min(token(expression, 0)), 1u);
    EXPECT_EQ(repeat_max(token(expression, 0)), 1u);
}

TEST(PathExpression, ContainsTransmission) {
    const PathExpression expression = compile_path_expression(".*<T>.*");
    ASSERT_TRUE(expression.valid);
    ASSERT_EQ(expression.token_count, 3u);
    EXPECT_EQ(scattering(token(expression, 0)), PATH_DEBUG_CLS_ANY);
    EXPECT_EQ(repeat_min(token(expression, 0)), 0u);
    EXPECT_EQ(repeat_max(token(expression, 0)), PATH_DEBUG_REPEAT_INF);
    EXPECT_EQ(side(token(expression, 1)), PATH_DEBUG_SIDE_TRANSMIT);
    EXPECT_EQ(scattering(token(expression, 1)), PATH_DEBUG_CLS_ANY);
}

TEST(PathExpression, Alternatives) {
    const PathExpression expression = compile_path_expression("[SG]+");
    ASSERT_TRUE(expression.valid);
    ASSERT_EQ(expression.token_count, 1u);
    EXPECT_EQ(scattering(token(expression, 0)), PATH_DEBUG_CLS_DELTA | PATH_DEBUG_CLS_GLOSSY);
    EXPECT_EQ(side(token(expression, 0)), PATH_DEBUG_SIDE_REFLECT | PATH_DEBUG_SIDE_TRANSMIT);
    EXPECT_EQ(repeat_min(token(expression, 0)), 1u);
    EXPECT_EQ(repeat_max(token(expression, 0)), PATH_DEBUG_REPEAT_INF);
}

TEST(PathExpression, BoundedRepeats) {
    const PathExpression exact = compile_path_expression(".{3}");
    ASSERT_TRUE(exact.valid);
    EXPECT_EQ(repeat_min(token(exact, 0)), 3u);
    EXPECT_EQ(repeat_max(token(exact, 0)), 3u);

    const PathExpression open = compile_path_expression("D{2,}");
    ASSERT_TRUE(open.valid);
    EXPECT_EQ(repeat_min(token(open, 0)), 2u);
    EXPECT_EQ(repeat_max(token(open, 0)), PATH_DEBUG_REPEAT_INF);

    const PathExpression optional = compile_path_expression("G?");
    ASSERT_TRUE(optional.valid);
    EXPECT_EQ(repeat_min(token(optional, 0)), 0u);
    EXPECT_EQ(repeat_max(token(optional, 0)), 1u);
}

TEST(PathExpression, QuotedCodesNameTechniques) {
    const PathExpression nee = compile_path_expression(".'n'");
    ASSERT_TRUE(nee.valid);
    ASSERT_EQ(nee.token_count, 2u);
    EXPECT_EQ(labels(token(nee, 0)), PATH_DEBUG_LABEL_ANY);
    EXPECT_EQ(labels(token(nee, 1)), 1u << PATH_RECORD_METHOD_LIGHT);
    EXPECT_EQ(scattering(token(nee, 1)), PATH_DEBUG_CLS_ANY);

    const PathExpression combined = compile_path_expression("<D'g'>+");
    ASSERT_TRUE(combined.valid);
    ASSERT_EQ(combined.token_count, 1u);
    EXPECT_EQ(scattering(token(combined, 0)), PATH_DEBUG_CLS_DIFFUSE);
    EXPECT_EQ(labels(token(combined, 0)), 1u << PATH_RECORD_METHOD_GUIDING);
    EXPECT_EQ(repeat_max(token(combined, 0)), PATH_DEBUG_REPEAT_INF);

    // outside brackets every symbol is an event of its own
    const PathExpression two = compile_path_expression("D'g'");
    ASSERT_TRUE(two.valid);
    EXPECT_EQ(two.token_count, 2u);

    const PathExpression either = compile_path_expression("['rrb''rrn''rec']");
    ASSERT_TRUE(either.valid);
    EXPECT_EQ(labels(token(either, 0)), (1u << PATH_RECORD_METHOD_REPLAYED_BSDF) |
                                            (1u << PATH_RECORD_METHOD_REPLAYED_LIGHT) |
                                            (1u << PATH_RECORD_METHOD_RECONNECTION));

    EXPECT_FALSE(compile_path_expression("'r'").valid);
    EXPECT_FALSE(compile_path_expression("'B'").valid);
    EXPECT_FALSE(compile_path_expression("'g").valid);
    EXPECT_FALSE(compile_path_expression("''").valid);
}

TEST(PathExpression, LightEndsTheExpression) {
    const PathExpression direct = compile_path_expression(".L");
    ASSERT_TRUE(direct.valid);
    EXPECT_EQ(direct.token_count, 1u);

    EXPECT_FALSE(compile_path_expression("L.").valid);
    EXPECT_FALSE(compile_path_expression("<DL>").valid);
    EXPECT_FALSE(compile_path_expression("<L'n'>").valid);
    EXPECT_FALSE(compile_path_expression(".N").valid);
}

TEST(PathExpression, RejectsWhatDoesNotParse) {
    EXPECT_FALSE(compile_path_expression("<X>").valid);
    EXPECT_FALSE(compile_path_expression("<RD").valid);
    EXPECT_FALSE(compile_path_expression("D{3,1}").valid);
    EXPECT_FALSE(compile_path_expression("D{8}").valid);
    EXPECT_FALSE(compile_path_expression("D{2,x}").valid);
    EXPECT_FALSE(compile_path_expression(".........").valid);
}

TEST(PathExpression, RecordedPathRoundTrips) {
    const uint32_t reflect_diffuse = PATH_RECORD_LOBE_DIFFUSE;
    const uint32_t transmit_glossy = PATH_RECORD_LOBE_GLOSSY | PATH_RECORD_LOBE_TRANSMISSION;
    EXPECT_EQ(path_event_name(reflect_diffuse), "RD");
    EXPECT_EQ(path_event_name(transmit_glossy), "TG");
    EXPECT_EQ(path_event_name(0), "RU");

    const uint32_t classes = reflect_diffuse | (transmit_glossy << 4);
    EXPECT_EQ(path_expression_of(classes, 2), "C<RD><TG><L.>");

    const PathExpression expression = compile_path_expression(path_expression_of(classes, 2));
    ASSERT_TRUE(expression.valid);
    ASSERT_EQ(expression.token_count, 2u);
    EXPECT_EQ(side(token(expression, 1)), PATH_DEBUG_SIDE_TRANSMIT);
    EXPECT_EQ(scattering(token(expression, 1)), PATH_DEBUG_CLS_GLOSSY);

    EXPECT_EQ(path_expression_of(classes, 12).substr(0, 13), "C<RD><TG><RU>");
    EXPECT_TRUE(path_expression_of(classes, 12).ends_with(".*"));
}
