#include "../include/merian/vk/shader/slang_entry_point.hpp"

#include <gtest/gtest.h>

using namespace merian;
using namespace Slang;

TEST(MathTest, Addition) {
    spdlog::set_level(spdlog::level::debug);

    merian::ShaderCompileContextHandle compilation_session_desc = merian::ShaderCompileContext::createTestContext();

    compilation_session_desc->add_search_path("/home/oschdi/Projects/merian-shadertoy/subprojects/merian/include/merian-shaders");
    compilation_session_desc->add_search_path("/home/oschdi/Projects/merian-shadertoy/subprojects/merian/tests");
    compilation_session_desc->add_search_path("/home/oschdi/Projects/merian-shadertoy/subprojects/merian/include/merian-shaders/utils");

    SlangSessionHandle session = merian::SlangSession::create(compilation_session_desc);

    // TODO allow SlandComposition to add Entry Points that are later found by findAndCheckEntryPoint()
    /*SlangCompositionHandle composition = SlangComposition::create();
    composition->add_module_from_path("exposure.slang", true);
    ComPtr<slang::IComponentType> composedProgram = session->compose(composition);*/

    SlangComposition::SlangModule slang_module = SlangComposition::SlangModule::from_path("hash.slang", false);
    auto module = session->load_module_from_source(
                                      slang_module.get_name(),
                                      slang_module.get_source(compilation_session_desc->get_search_path_file_loader()),
                                    slang_module.get_import_path());

    ComPtr<slang::IEntryPoint> entry_point;
    {
        ComPtr<slang::IBlob> diagnostics_blob;
        module->findAndCheckEntryPoint("merian.pcg3d", SLANG_STAGE_CALLABLE, entry_point.writeRef(), diagnostics_blob.writeRef());
        if (diagnostics_blob != nullptr) {
            SPDLOG_INFO("Slang find entry point. Diagnostics: {}",
                         merian::SlangSession::diagnostics_as_string(diagnostics_blob));
        }
        SPDLOG_INFO("Found and validated entry point {}", entry_point->getFunctionReflection()->getName());
    }

    std::vector<slang::IComponentType*> components;
    components.push_back(module);
    components.push_back(entry_point);

    ComPtr<slang::IComponentType> composition = session->compose(components);

    ComPtr<ISlangSharedLibrary> sharedLibrary;
    {
        ComPtr<slang::IBlob> diagnostics_blob;
        SlangResult result = composition->getEntryPointHostCallable(
            0,
            0,
            sharedLibrary.writeRef(),
            diagnostics_blob.writeRef());
        if (diagnostics_blob != nullptr) {
            SPDLOG_DEBUG("Slang composing components. Diagnostics: {}",
                         merian::SlangSession::diagnostics_as_string(diagnostics_blob));
        }
    }

    EXPECT_EQ(2 + 2, 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}