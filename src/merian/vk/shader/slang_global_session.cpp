#include "merian/vk/shader/slang_global_session.hpp"

namespace {
    Slang::ComPtr<slang::IGlobalSession> global_session;
}

namespace merian {

Slang::ComPtr<slang::IGlobalSession> get_global_slang_session() {
    if (global_session.get() == nullptr) {
        createGlobalSession(global_session.writeRef());
    }
    return global_session;
}

}
