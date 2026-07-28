#include "merian/utils/dynamic_library.hpp"

#include <fmt/format.h>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace merian {

#if defined(_WIN32)

DynamicLibrary::DynamicLibrary(const std::filesystem::path& path) : path(path) {
    // dependencies resolve from the library's own dir, the exe dir, and PATH (meson devenv)
    handle = LoadLibraryExW(path.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (handle == nullptr) {
        throw load_failed{
            fmt::format("could not load '{}' (error {})", path.string(), GetLastError())};
    }
}

DynamicLibrary::~DynamicLibrary() {
    if (handle != nullptr) {
        FreeLibrary(static_cast<HMODULE>(handle));
    }
}

void* DynamicLibrary::get_symbol(const std::string& name) const {
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name.c_str()));
}

#else

DynamicLibrary::DynamicLibrary(const std::filesystem::path& path) : path(path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* err = dlerror();
        throw load_failed{fmt::format("could not load '{}' ({})", path.string(),
                                      err != nullptr ? err : "unknown error")};
    }
}

DynamicLibrary::~DynamicLibrary() {
    if (handle != nullptr) {
        dlclose(handle);
    }
}

void* DynamicLibrary::get_symbol(const std::string& name) const {
    return dlsym(handle, name.c_str());
}

#endif

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : path(std::move(other.path)), handle(std::exchange(other.handle, nullptr)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        std::swap(path, other.path);
        std::swap(handle, other.handle);
    }
    return *this;
}

} // namespace merian
