#include "merian/utils/renderdoc.hpp"

#ifdef RENDERDOC_AVAILABLE
#include "renderdoc_app.h"

#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <dlfcn.h>
#endif

static RENDERDOC_API_1_1_2* rdoc_api = NULL;
#endif

namespace merian {

Renderdoc::Renderdoc() {
#ifdef RENDERDOC_AVAILABLE
    pRENDERDOC_GetAPI RENDERDOC_GetAPI = nullptr;

#ifdef _WIN32
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
        RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
#elif __linux__
    if (void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
        RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
#endif

    if (RENDERDOC_GetAPI != nullptr) {
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void**)&rdoc_api);
        assert(ret == 1);
    }
#endif
}

void Renderdoc::start_frame_capture() {
#ifdef RENDERDOC_AVAILABLE
    if (rdoc_api)
        rdoc_api->StartFrameCapture(NULL, NULL);
#endif
}
void Renderdoc::end_frame_capture() {
#ifdef RENDERDOC_AVAILABLE
    if (rdoc_api)
        rdoc_api->EndFrameCapture(NULL, NULL);
#endif
}

} // namespace merian
