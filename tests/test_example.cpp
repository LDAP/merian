#include "../include/merian/vk/shader/slang_entry_point.hpp"
#include "../include/merian/vk/shader/slang_test.hpp"

#include <gtest/gtest.h>

using namespace merian;
using namespace Slang;

TEST(MathTest, Addition) {
    spdlog::set_level(spdlog::level::debug);

    std::shared_ptr<merian_nodes::SlangTester> test_compiler = std::make_shared<merian_nodes::SlangTester>();

    merian::ShaderCompileContextHandle compilation_session_desc = merian::ShaderCompileContext::createTestContext();

    compilation_session_desc->add_search_path("/home/oschdi/Projects/merian-shadertoy/subprojects/merian/include/merian-shaders/utils");

    SlangSessionHandle session = merian::SlangSession::create(compilation_session_desc);
    SlangCompositionHandle composition = SlangComposition::create();
    composition->add_module_from_path("encoding.slang", true);
    //composition->add_entry_point("decode_normal", "encoding");

    SlangComposition::SlangModule slang_module = SlangComposition::SlangModule::from_path("encoding.slang", false);
    auto module = session->load_module_from_source(
                                      slang_module.get_name(),
                                      slang_module.get_source(compilation_session_desc->get_search_path_file_loader()),
                                    slang_module.get_import_path());

    ComPtr<slang::IEntryPoint> entry_point;
    {
        ComPtr<slang::IBlob> diagnostics_blob;
        module->findAndCheckEntryPoint("decode_normal", SlangStage::SLANG_STAGE_CALLABLE, entry_point.writeRef(), diagnostics_blob.writeRef());
        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang composing components. Diagnostics: {}",
                         merian::SlangSession::diagnostics_as_string(diagnostics_blob));
        }

    }
    ComPtr<slang::IComponentType> composedProgram = session->compose(composition);
    /*ComPtr<ISlangSharedLibrary> sharedLibrary;
    {
        ComPtr<slang::IBlob> diagnostics_blob;
        SlangResult result = composedProgram->getEntryPointHostCallable(
            0,
            0,
            sharedLibrary.writeRef(),
            diagnostics_blob.writeRef());
        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang composing components. Diagnostics: {}",
                         merian::SlangSession::diagnostics_as_string(diagnostics_blob));
        }
    }*/

    EXPECT_EQ(2 + 2, 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}