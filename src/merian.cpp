#include "vk/extension/extension_debug_utils.hpp"
#include "vk/extension/extension_float_atomics.hpp"
#include "vk/extension/extension_glfw.hpp"
#include "vk/extension/extension_raytrace.hpp"
#include "vk/extension/extension_v12.hpp"
#include <iostream>
#include <spdlog/spdlog.h>
#include <vk/context.hpp>

void setup_logging() {
#ifdef DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif
}

int main(int argc, char** argv) {
    if (argc != 1) {
        std::cout << argv[0] << "takes no arguments.\n";
        return 1;
    }
    std::cout << "This is " << PROJECT_NAME << " " << VERSION << ".\n";

    setup_logging();

    std::vector<Extension*> extensions;

#ifdef DEBUG
    extensions.push_back(new ExtensionDebugUtils());
#endif
    extensions.push_back(new ExtensionGLFW());
    extensions.push_back(new ExtensionRaytraceQuery());
    extensions.push_back(new ExtensionV12());
    extensions.push_back(new ExtensionFloatAtomics());
    {
        Context context(extensions);
    }

    for (auto& ext : extensions) {
        delete ext;
    }

    return 0;
}
