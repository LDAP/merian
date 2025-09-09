#include "../include/merian/vk/shader/slang_entry_point.hpp"
#include "../include/merian/vk/shader/slang_test.hpp"

#include <gtest/gtest.h>

TEST(MathTest, Addition) {
    std::shared_ptr<merian_nodes::SlangTester> test_compiler = std::make_shared<merian_nodes::SlangTester>();

    merian::ShaderCompileContextHandle compilation_session_desc = merian::ShaderCompileContext::create();

    compilation_session_desc->add_search_path("merian-nodes/nodes/svgf");

    merian::EntryPointHandle module =
    merian::SlangProgramEntryPoint::create(compilation_session_desc, "svgf_filter.slang");
    EXPECT_EQ(2 + 2, 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}