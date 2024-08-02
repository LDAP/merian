#pragma once

inline void debugbreak() {

#if defined(_MSVC_LANG)
    __debugbreak();
#elif defined(LINUX)
    raise(SIGTRAP);
#endif

}

