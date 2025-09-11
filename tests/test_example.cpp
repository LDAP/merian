#include "../include/merian/vk/shader/slang_entry_point.hpp"
#include "../include/merian/vk/shader/slang_test.hpp"

#include <gtest/gtest.h>

TEST(MathTest, Addition) {
    std::shared_ptr<merian_nodes::SlangTester> test_compiler = std::make_shared<merian_nodes::SlangTester>();

    merian::ShaderCompileContextHandle compilation_session_desc = merian::ShaderCompileContext::create();
    SPDLOG_DEBUG("added search path");

    compilation_session_desc->add_search_path("/home/oschdi/Projects/merian-shadertoy/subprojects/merian/include/merian-shaders/utils");

    merian::SlangSessionHandle session = merian::SlangSession::create(compilation_session_desc);
    merian::SlangComposition::SlangModule slang_module = merian::SlangComposition::SlangModule::from_path("encoding.slang", false);
    Slang::ComPtr<slang::IModule> module = session->load_module_from_source(
        slang_module.get_name(),
        slang_module.get_source(compilation_session_desc->get_search_path_file_loader()),
        slang_module.get_import_path());
    EXPECT_EQ(2 + 2, 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}