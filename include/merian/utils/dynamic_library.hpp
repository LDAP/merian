#pragma once

#include <filesystem>
#include <string>

namespace merian {

// RAII wrapper around a dynamically loaded shared library (dlopen / LoadLibrary).
// The library stays loaded for the lifetime of this object.
class DynamicLibrary {
  public:
    class load_failed : public std::runtime_error {
      public:
        load_failed(const std::string& what) : std::runtime_error(what) {}
    };

    // Loads the library at path. Throws load_failed on error.
    explicit DynamicLibrary(const std::filesystem::path& path);

    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    DynamicLibrary(DynamicLibrary&& other) noexcept;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

    ~DynamicLibrary();

    // Resolves an exported symbol, or nullptr if it is not present.
    void* get_symbol(const std::string& name) const;

    // Typed convenience over get_symbol.
    template <typename FuncPtr> FuncPtr get_symbol(const std::string& name) const {
        return reinterpret_cast<FuncPtr>(get_symbol(name));
    }

    const std::filesystem::path& get_path() const {
        return path;
    }

  private:
    std::filesystem::path path;
    void* handle = nullptr;
};

} // namespace merian
