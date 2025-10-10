#include "../include/merian/vk/shader/slang_entry_point.hpp"
#include "merian/vk/shader/slang_shared_library.hpp"
#include "slang-com-helper.h"
#include "slang.h"
#include <slang-com-ptr.h>

#include <glm/vec3.hpp>
#include <gtest/gtest.h>

using namespace merian;

TEST(MathTest, StackingVectors) {
    spdlog::set_level(spdlog::level::debug);
    SlangSharedLibraryHandle library = SlangSharedLibrary::create(
        "/home/oschdi/Projects/merian-shadertoy/subprojects/merian/tests/test.slang");

    typedef glm::uvec3 (*Func)(uint32_t, uint32_t, uint32_t);
    const auto func = library->getFunctionByName<Func>("stack_to_vec");

    glm::uvec3 result = func(1, 2, 3);
    EXPECT_EQ(result, glm::uvec3(1, 2, 3));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}