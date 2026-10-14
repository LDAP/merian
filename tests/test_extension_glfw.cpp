#include <gtest/gtest.h>

#include "merian/vk/context.hpp"
#include "merian/vk/extension/glfw/extension_glfw.hpp"

#include <cstdlib>

using namespace merian;

// Without a display GLFW fails to initialize: the context is created without the extension, and
// GLFW errors still throw for the callers that handle them.
TEST(ExtensionGLFWTest, FailedInitializationKeepsErrorsThrowing) {
    unsetenv("WAYLAND_DISPLAY");
    unsetenv("DISPLAY");

    const ContextHandle context = Context::create({
        .context_extensions = {ExtensionGLFW::name},
        .application_name = "test-extension-glfw",
    });
    EXPECT_FALSE(context->get_context_extension<ExtensionGLFW>(true));
    EXPECT_THROW(glfwWindowHint(-1, 0), ExtensionGLFW::glfw_error);
}
