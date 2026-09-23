#include <gtest/gtest.h>

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/path_expression.hpp"

using namespace merian;

namespace {

uint32_t token(const PathExpression& expression, const uint32_t index) {
    return (expression.words[index / 2] >> (16 * (index % 2))) & 0xFFFFu;
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

} // namespace

TEST(PathExpression, EmptyAcceptsEverything) {
    const PathExpression expression = compile_path_expression("");
    EXPECT_TRUE(expression.valid);
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

TEST(PathExpression, TrailingNMatchesNextEventConnections) {
    const PathExpression plain = compile_path_expression("CD+");
    EXPECT_FALSE(plain.nee);

    const PathExpression direct = compile_path_expression("CDN");
    ASSERT_TRUE(direct.valid);
    EXPECT_TRUE(direct.nee);
    ASSERT_EQ(direct.token_count, 1u);
    EXPECT_EQ(scattering(token(direct, 0)), PATH_DEBUG_CLS_DIFFUSE);

    const PathExpression any = compile_path_expression("N");
    ASSERT_TRUE(any.valid);
    EXPECT_TRUE(any.nee);
    ASSERT_EQ(any.token_count, 1u);
    EXPECT_EQ(repeat_min(token(any, 0)), 0u);
    EXPECT_EQ(repeat_max(token(any, 0)), PATH_DEBUG_REPEAT_INF);

    EXPECT_FALSE(compile_path_expression("NDD").valid);
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
