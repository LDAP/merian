#pragma once

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/utils/hash.hpp"

#include "slang-com-ptr.h"
#include "slang.h"
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace merian {

class SlangProgram;
using SlangProgramHandle = std::shared_ptr<SlangProgram>;

class SlangSession;
using SlangSessionHandle = std::shared_ptr<SlangSession>;

class SlangSession {
  protected:
    SlangSession(const ShaderCompileContextHandle& shader_compile_context);

  public:
    Slang::ComPtr<slang::IComponentType> compose(const SlangCompositionHandle& composition);

    static Slang::ComPtr<slang::IComponentType>
    link(const Slang::ComPtr<slang::IComponentType>& composed_program);

    static Slang::ComPtr<slang::IBlob>
    compile(const Slang::ComPtr<slang::IComponentType>& linked_program, uint32_t entry_point_index);

    // Compiles all entry points of the linked program.
    static Slang::ComPtr<slang::IBlob>
    compile(const Slang::ComPtr<slang::IComponentType>& linked_program);

    // Caller must keep the program alive to keep the type layout valid.
    struct TypeLayoutResult {
        slang::TypeLayoutReflection* type_layout;
        SlangProgramHandle program;
    };

    static TypeLayoutResult get_type_layout(const ShaderCompileContextHandle& compile_context,
                                            const SlangCompositionHandle& composition,
                                            const std::string& type_name);

  public:
    // Thrown when a composition needs a fresh session.
    class stale_session : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    static SlangSessionHandle create(const ShaderCompileContextHandle& shader_compile_context);

    ~SlangSession();

  private:
    // The path is the identity for path-based imports of the module.
    Slang::ComPtr<slang::IModule>
    load_module_from_source(const std::string& name,
                            const std::string& source,
                            const std::optional<std::filesystem::path>& path);

    // Both types must be known to the component. "id" is the createDynamicObject<>(id, ...)
    // dispatch id, -1 lets the compiler choose.
    Slang::ComPtr<slang::ITypeConformance>
    create_type_conformance(const Slang::ComPtr<slang::IComponentType>& component,
                            const std::string& type_name,
                            const std::string& interface_name,
                            int64_t id);

    Slang::ComPtr<slang::IComponentType>
    compose_components(const std::vector<Slang::ComPtr<slang::IComponentType>>& components);

    // --- on-disk shader cache (serialized IR modules + SPIR-V) ---

    // MERIAN_SHADER_CACHE != "0".
    static bool cache_enabled();
    // MERIAN_SHADER_CACHE_DIR, else <cwd>/.merian-cache.
    static const std::filesystem::path& cache_root();
    // Sanitized Slang build tag; the level under the root that isolates Slang versions.
    static const std::string& cache_tag();
    // <root>/<slang-build-tag>/<subdir>; the build-tag level isolates Slang versions.
    static std::filesystem::path cache_dir(std::string_view subdir);
    // File -> owning blob (nullptr if absent/unreadable); touches mtime on a hit.
    static Slang::ComPtr<slang::IBlob> cache_read(const std::filesystem::path& path);
    // Atomic write (temp + rename), best-effort.
    static void cache_write(const std::filesystem::path& path, const void* data, size_t size);
    // Drops other build tags, then an LRU size-cap sweep (MERIAN_SHADER_CACHE_MAX_MB,
    // default 512, 0 = unbounded).
    static void cache_evict();

    // One directory per compile configuration.
    static std::filesystem::path module_cache_dir(const ShaderCompileContext& context);
    // Presents cached binary modules as `.slang-module` files beside their sources.
    static Slang::ComPtr<ISlangFileSystemExt>
    create_file_system(const std::filesystem::path& module_cache_dir,
                       std::function<bool()> serve_binaries);
    // Writes the binaries of the file modules this session compiled.
    void store_modules();
    std::optional<std::filesystem::path>
    canonical_file_path(const std::filesystem::path& path) const;
    // The name of the file's module when imported by its canonical spelling, nullopt outside the
    // search paths.
    std::optional<std::string> canonical_module_name(const std::filesystem::path& file) const;
    // Slang names a module after the first import that reaches it, and mangled names contain the
    // module name: a binary cannot load against a dependency compiled under another name.
    bool modules_canonically_named();
    // Slang's own backend cache key (getEntryPointHash) as a filename, or nullopt to skip.
    static std::optional<std::filesystem::path>
    spirv_cache_path(const Slang::ComPtr<slang::IComponentType>& program);
    static std::optional<std::filesystem::path>
    spirv_cache_path(const Slang::ComPtr<slang::IComponentType>& program,
                     uint32_t entry_point_index);

    const Slang::ComPtr<slang::IModule>& ensure_module(SlangComposition::SlangModule& module);
    static void check_one_source_per_name(const SlangCompositionHandle& composition,
                                          std::unordered_map<std::string, uint64_t>& sources);
    Slang::ComPtr<slang::IComponentType> compose_tree(const SlangCompositionHandle& composition);
    bool is_imported(slang::IModule* module) const;
    bool is_replaced(slang::IModule* module) const;
    void check_no_replaced_dependency(slang::IModule* module) const;

  private:
    // The context owns the session it hands out, so the way back must not own it: a handle here
    // closes a reference cycle that keeps both alive past the last use, and with them the
    // destructor that caps the cache on disk.
    ShaderCompileContextHandle compile_context() const {
        return weak_compile_context.lock();
    }

    static constexpr std::size_t MAX_REPLACED_MODULES = 64;

    using ModuleVersion = std::pair<std::string, uint64_t>;
    struct ModuleVersionHash {
        std::size_t operator()(const ModuleVersion& version) const {
            return hash_val(version.first, version.second);
        }
    };

    using ConformanceKey = std::pair<SlangComposition::TypeConformance, int64_t>;
    struct ConformanceKeyHash {
        std::size_t operator()(const ConformanceKey& key) const {
            return hash_val(SlangComposition::TypeConformance::HashFunction{}(key.first),
                            key.second);
        }
    };

    const std::weak_ptr<ShaderCompileContext> weak_compile_context;
    Slang::ComPtr<ISlangFileSystemExt> file_system;
    Slang::ComPtr<slang::ISession> session;

    // -> renamed entry point
    std::unordered_map<SlangComposition::EntryPoint,
                       Slang::ComPtr<slang::IComponentType>,
                       SlangComposition::EntryPoint::HashFunction>
        entry_point_cache;
    // identity (the path Slang reports as dependency) -> module
    std::unordered_map<std::string, slang::IModule*> source_strings_by_identity;
    std::unordered_set<slang::IModule*> checked_module_names;
    bool module_names_canonical = true;
    struct BoundModule {
        uint64_t source_hash;
        Slang::ComPtr<slang::IModule> module;
    };
    std::unordered_map<std::string, BoundModule> bound_modules;
    std::unordered_map<ModuleVersion, Slang::ComPtr<slang::IModule>, ModuleVersionHash>
        module_versions;
    std::unordered_map<ConformanceKey, Slang::ComPtr<slang::IComponentType>, ConformanceKeyHash>
        type_conformance_cache;
    // -> composition version, composed
    std::unordered_map<SlangCompositionHandle,
                       std::pair<uint64_t, Slang::ComPtr<slang::IComponentType>>>
        composition_cache;
};

} // namespace merian
