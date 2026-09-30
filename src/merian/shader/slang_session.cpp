#include "merian/shader/slang_session.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/hash.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace merian {

namespace {

std::string to_hex(const void* data, const size_t size) {
    static constexpr std::string_view digits = "0123456789abcdef";
    const auto* bytes = static_cast<const uint8_t*>(data);
    std::string out(size * 2, '0');
    for (size_t i = 0; i < size; i++) {
        out[2 * i] = digits[bytes[i] >> 4];
        out[(2 * i) + 1] = digits[bytes[i] & 0x0F];
    }
    return out;
}

double milliseconds_since(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

std::string entry_point_names(const Slang::ComPtr<slang::IComponentType>& program) {
    slang::ProgramLayout* layout = program->getLayout();
    std::string names;
    for (SlangUInt i = 0; i < layout->getEntryPointCount(); i++) {
        names += fmt::format("{}{}", i == 0 ? "" : ", ",
                             layout->getEntryPointByIndex(i)->getNameOverride());
    }
    return names;
}

// Minimal owning ISlangBlob over a byte buffer, so cached files can be handed to
// loadModuleFromIRBlob and to SPIR-V consumers.
class CacheBlob final : public ISlangBlob {
  public:
    explicit CacheBlob(std::vector<std::byte>&& data) : data(std::move(data)) {}

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid,
                                                          void** out_object) override {
        static constexpr SlangUUID unknown_guid = ISlangUnknown::getTypeGuid();
        static constexpr SlangUUID blob_guid = ISlangBlob::getTypeGuid();
        if (std::memcmp(&uuid, &unknown_guid, sizeof(SlangUUID)) == 0 ||
            std::memcmp(&uuid, &blob_guid, sizeof(SlangUUID)) == 0) {
            addRef();
            *out_object = static_cast<ISlangBlob*>(this);
            return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override {
        return ++ref_count;
    }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override {
        const uint32_t rc = --ref_count;
        if (rc == 0) {
            delete this;
        }
        return rc;
    }

    SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() override {
        return data.data();
    }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() override {
        return data.size();
    }

  private:
    std::vector<std::byte> data;
    std::atomic<uint32_t> ref_count{1};
};

// The file Slang resolves `import <name>` to through the search paths.
std::optional<std::filesystem::path>
resolve_module(const std::string& name, const std::vector<std::filesystem::path>& search_paths) {
    std::vector<std::string> file_names;
    if (name.ends_with(".slang")) {
        file_names = {name};
    } else {
        std::string dashed = name;
        std::ranges::replace(dashed, '_', '-');
        file_names = {name + ".slang", dashed + ".slang"};
    }
    for (const std::string& file_name : file_names) {
        for (const std::filesystem::path& search_path : search_paths) {
            std::error_code ec;
            const std::filesystem::path candidate = search_path / file_name;
            if (std::filesystem::is_regular_file(candidate, ec)) {
                const std::filesystem::path canonical =
                    std::filesystem::weakly_canonical(candidate, ec);
                return ec ? candidate : canonical;
            }
        }
    }
    return std::nullopt;
}

// `import a.b` names the module `a/b`, `import "a/b.slang"` names it `a/b.slang`.
std::string module_name_of_import(const std::string& spelling) {
    if (spelling.starts_with('"')) {
        return spelling.substr(1, spelling.size() - 2);
    }
    std::string name = spelling;
    std::ranges::replace(name, '.', '/');
    return name;
}

std::string import_of_module_name(const std::string& module_name) {
    if (module_name.ends_with(".slang")) {
        return fmt::format("\"{}\"", module_name);
    }
    std::string dotted = module_name;
    std::ranges::replace(dotted, '/', '.');
    return dotted;
}

Slang::ComPtr<ISlangBlob> string_blob(const std::string& string) {
    std::vector<std::byte> data(string.size() + 1);
    std::memcpy(data.data(), string.c_str(), string.size() + 1);
    Slang::ComPtr<ISlangBlob> blob;
    blob.attach(new CacheBlob(std::move(data)));
    return blob;
}

// The OS file system, serving a cached binary module as `<source>.slang-module` beside its source.
class ModuleFileSystem final : public ISlangFileSystemExt {
  public:
    ModuleFileSystem(std::filesystem::path cache_directory, std::function<bool()> serve_binaries)
        : cache_directory(std::move(cache_directory)), serve_binaries(std::move(serve_binaries)) {}

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid,
                                                          void** out_object) override {
        if (void* object = castAs(uuid)) {
            addRef();
            *out_object = object;
            return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override {
        return ++ref_count;
    }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override {
        const uint32_t rc = --ref_count;
        if (rc == 0) {
            delete this;
        }
        return rc;
    }
    SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& uuid) override {
        const auto is = [&](const SlangUUID& other) {
            return std::memcmp(&uuid, &other, sizeof(SlangUUID)) == 0;
        };
        if (is(ISlangUnknown::getTypeGuid()) || is(ISlangCastable::getTypeGuid()) ||
            is(ISlangFileSystem::getTypeGuid()) || is(ISlangFileSystemExt::getTypeGuid())) {
            return static_cast<ISlangFileSystemExt*>(this);
        }
        return nullptr;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(char const* path,
                                                    ISlangBlob** out_blob) override {
        const std::filesystem::path file = resolve(path);
        std::error_code ec;
        const auto size = std::filesystem::file_size(file, ec);
        std::ifstream in(file, std::ios::binary);
        if (ec || !in) {
            return SLANG_E_NOT_FOUND;
        }
        std::vector<std::byte> data(static_cast<size_t>(size));
        if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
            return SLANG_FAIL;
        }
        if (file != path) {
            served_binaries.insert(file.string());
            // youngest for eviction, as cache_read does
            std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now(),
                                             ec);
        }
        *out_blob = new CacheBlob(std::move(data));
        return SLANG_OK;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    getFileUniqueIdentity(const char* path, ISlangBlob** out_unique_identity) override {
        std::error_code ec;
        const std::filesystem::path canonical = std::filesystem::weakly_canonical(path, ec);
        if (ec) {
            return SLANG_FAIL;
        }
        *out_unique_identity = string_blob(canonical.string()).detach();
        return SLANG_OK;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(SlangPathType from_path_type,
                                                            const char* from_path,
                                                            const char* path,
                                                            ISlangBlob** path_out) override {
        const std::filesystem::path from{from_path};
        const std::filesystem::path directory =
            from_path_type == SLANG_PATH_TYPE_FILE ? from.parent_path() : from;
        *path_out = string_blob((directory / path).string()).detach();
        return SLANG_OK;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getPathType(const char* path,
                                                       SlangPathType* path_type_out) override {
        std::error_code ec;
        const std::filesystem::file_status status = std::filesystem::status(resolve(path), ec);
        if (ec || !std::filesystem::exists(status)) {
            return SLANG_E_NOT_FOUND;
        }
        *path_type_out = std::filesystem::is_directory(status) ? SLANG_PATH_TYPE_DIRECTORY
                                                               : SLANG_PATH_TYPE_FILE;
        return SLANG_OK;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getPath(PathKind kind,
                                                   const char* path,
                                                   ISlangBlob** out_path) override {
        switch (kind) {
        case PathKind::Simplified:
            *out_path =
                string_blob(std::filesystem::path{path}.lexically_normal().string()).detach();
            return SLANG_OK;
        case PathKind::Canonical: {
            std::error_code ec;
            const std::filesystem::path canonical = std::filesystem::weakly_canonical(path, ec);
            if (ec) {
                return SLANG_FAIL;
            }
            *out_path = string_blob(canonical.string()).detach();
            return SLANG_OK;
        }
        case PathKind::OperatingSystem:
        case PathKind::Display:
            *out_path = string_blob(path).detach();
            return SLANG_OK;
        default:
            return SLANG_E_NOT_IMPLEMENTED;
        }
    }

    SLANG_NO_THROW void SLANG_MCALL clearCache() override {}

    SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(const char*,
                                                                 FileSystemContentsCallBack,
                                                                 void*) override {
        return SLANG_E_NOT_IMPLEMENTED;
    }

    SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() override {
        return OSPathKind::Direct;
    }

    std::filesystem::path binary_path(const std::filesystem::path& source) const {
        std::error_code ec;
        const std::filesystem::path canonical = std::filesystem::weakly_canonical(source, ec);
        return cache_directory /
               fmt::format("{:016x}.slang-module", hash_val((ec ? source : canonical).string()));
    }

    // A binary written while a session is open would load as a second module of its source.
    void hide(const std::filesystem::path& binary) {
        hidden_binaries.insert(binary.string());
    }

    bool is_hidden(const std::filesystem::path& binary) const {
        return hidden_binaries.contains(binary.string());
    }

  private:
    std::filesystem::path resolve(const std::filesystem::path& path) const {
        if (path.extension() != ".slang-module") {
            return path;
        }
        std::filesystem::path source = path;
        source.replace_extension(".slang");
        std::error_code ec;
        if (!std::filesystem::exists(source, ec)) {
            return path;
        }
        const std::filesystem::path binary = binary_path(source);
        if (is_hidden(binary)) {
            return path;
        }
        // a served binary stays served, or another import spelling compiles a second module
        return served_binaries.contains(binary.string()) || serve_binaries() ? binary : path;
    }

    const std::filesystem::path cache_directory;
    const std::function<bool()> serve_binaries;
    std::unordered_set<std::string> hidden_binaries;
    std::unordered_set<std::string> served_binaries;
    std::atomic<uint32_t> ref_count{0};
};

} // namespace

SlangSessionHandle SlangSession::create(const ShaderCompileContextHandle& shader_compile_context) {
    SPDLOG_DEBUG("create slang session");
    return SlangSessionHandle(new SlangSession(shader_compile_context));
}

SlangSession::~SlangSession() {
    cache_evict();
}

// --- on-disk shader cache ---

bool SlangSession::cache_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MERIAN_SHADER_CACHE");
        return env == nullptr || std::string_view{env} != "0";
    }();
    return enabled;
}

const std::filesystem::path& SlangSession::cache_root() {
    static const std::filesystem::path root = [] {
        if (const char* dir = std::getenv("MERIAN_SHADER_CACHE_DIR")) {
            return std::filesystem::path{dir};
        }
        std::error_code ec;
        const std::filesystem::path cwd = std::filesystem::current_path(ec);
        return (ec ? std::filesystem::path{"."} : cwd) / ".merian-cache";
    }();
    return root;
}

const std::string& SlangSession::cache_tag() {
    static const std::string tag = [] {
        std::string t = get_global_slang_session()->getBuildTagString();
        for (char& c : t) {
            if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '.' && c != '-' &&
                c != '_') {
                c = '_';
            }
        }
        return t;
    }();
    return tag;
}

std::filesystem::path SlangSession::cache_dir(const std::string_view subdir) {
    return cache_root() / cache_tag() / subdir;
}

Slang::ComPtr<slang::IBlob> SlangSession::cache_read(const std::filesystem::path& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size == 0) {
        return nullptr;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return nullptr;
    }
    std::vector<std::byte> data(static_cast<size_t>(size));
    if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
        return nullptr;
    }

    // touch mtime so frequently used entries stay youngest and survive eviction
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);

    Slang::ComPtr<slang::IBlob> blob;
    blob.attach(new CacheBlob(std::move(data)));
    return blob;
}

void SlangSession::cache_write(const std::filesystem::path& path,
                               const void* data,
                               const size_t size) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        return;
    }

    static const uint64_t salt = std::random_device{}();
    static std::atomic<uint64_t> counter{0};
    std::filesystem::path tmp = path;
    tmp += fmt::format(".tmp.{:x}.{}", salt, counter.fetch_add(1, std::memory_order_relaxed));

    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return;
        }
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!out) {
            out.close();
            std::filesystem::remove(tmp, ec);
            return;
        }
    }

    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
    }
}

void SlangSession::cache_evict() {
    if (!cache_enabled()) {
        return;
    }

    uint64_t budget = 512ull * 1024 * 1024;
    if (const char* env = std::getenv("MERIAN_SHADER_CACHE_MAX_MB")) {
        const uint64_t mb = std::strtoull(env, nullptr, 10);
        if (mb == 0) {
            return; // unbounded / manual
        }
        budget = mb * 1024ull * 1024;
    }

    std::error_code ec;
    const std::filesystem::path& root = cache_root();
    if (!std::filesystem::exists(root, ec) || ec) {
        return;
    }

    // Nothing under another build tag can be read again, so it goes before the size cap gets to
    // evict what this session just compiled.
    for (std::filesystem::directory_iterator it(root, ec), end; it != end; it.increment(ec)) {
        if (ec) {
            break;
        }
        std::error_code entry_ec;
        if (it->is_directory(entry_ec) && !entry_ec && it->path().filename() != cache_tag()) {
            std::filesystem::remove_all(it->path(), entry_ec);
        }
    }
    ec.clear();

    struct Entry {
        std::filesystem::path path;
        uint64_t size;
        std::filesystem::file_time_type mtime;
    };
    std::vector<Entry> entries;
    uint64_t total = 0;

    for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end;
         it.increment(ec)) {
        if (ec) {
            break;
        }
        if (!it->is_regular_file(ec) || ec) {
            continue;
        }
        const uint64_t size = it->file_size(ec);
        if (ec) {
            continue;
        }
        const auto mtime = it->last_write_time(ec);
        if (ec) {
            continue;
        }
        total += size;
        entries.push_back({it->path(), size, mtime});
    }

    if (total <= budget) {
        return;
    }

    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });

    for (const Entry& e : entries) {
        if (total <= budget) {
            break;
        }
        if (std::filesystem::remove(e.path, ec) && !ec) {
            total -= e.size;
        }
    }
}

Slang::ComPtr<slang::IModule>
SlangSession::load_module_from_source(const std::string& name,
                                      const std::string& source,
                                      const std::optional<std::filesystem::path>& path) {
    const auto start = std::chrono::steady_clock::now();
    Slang::ComPtr<slang::IBlob> diagnostics_blob;
    Slang::ComPtr<slang::IModule> module;

    const std::optional<std::filesystem::path> file =
        path ? canonical_file_path(*path) : std::nullopt;
    if (file && FileLoader::load_file_as_string(*file) == source) {
        // loaded like an import, so each file is one module under every spelling
        module = session->loadModule(
            canonical_module_name(*file).value_or(file->generic_string()).c_str(),
            diagnostics_blob.writeRef());
    } else {
        // the path is the identity: a source string's identity is otherwise its content
        const std::string identity = path ? path->generic_string() : name;
        module = session->loadModuleFromSourceString(name.c_str(), identity.c_str(), source.c_str(),
                                                     diagnostics_blob.writeRef());
        if (module != nullptr) {
            source_strings_by_identity.insert_or_assign(module->getFilePath(), module);
        }
    }

    if (module == nullptr) {
        throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
    }
    if (diagnostics_blob != nullptr) {
        SPDLOG_DEBUG("Slang loading module {}. Diagnostics: {}", name,
                     diagnostics_as_string(diagnostics_blob));
    }
    check_no_replaced_dependency(module);
    SPDLOG_INFO("slang: module {} {} ({:.1f} ms)", name,
                std::filesystem::path{module->getFilePath()}.extension() == ".slang-module"
                    ? "loaded from cache"
                    : "loaded",
                milliseconds_since(start));
    return module;
}

void SlangSession::store_modules() {
    if (!file_system || !modules_canonically_named()) {
        return;
    }
    auto* module_file_system = static_cast<ModuleFileSystem*>(file_system.get());
    for (SlangInt i = 0; i < session->getLoadedModuleCount(); i++) {
        slang::IModule* module = session->getLoadedModule(i);
        const char* path = module->getFilePath();
        if (path == nullptr || std::filesystem::path{path}.extension() != ".slang" ||
            source_strings_by_identity.contains(path)) {
            continue;
        }
        const std::optional<std::filesystem::path> source = canonical_file_path(path);
        // Slang cannot find a file outside the search paths again to validate its binary
        if (!source || !canonical_module_name(*source)) {
            continue;
        }
        const std::filesystem::path binary = module_file_system->binary_path(*source);
        if (module_file_system->is_hidden(binary)) {
            continue;
        }
        // a binary depending on a source string could never be proven up to date
        bool from_files = true;
        for (SlangInt32 j = 0; j < module->getDependencyFileCount() && from_files; j++) {
            from_files = canonical_file_path(module->getDependencyFilePath(j)).has_value();
        }
        Slang::ComPtr<slang::IBlob> blob;
        if (from_files && SLANG_SUCCEEDED(module->serialize(blob.writeRef())) && blob != nullptr) {
            cache_write(binary, blob->getBufferPointer(), blob->getBufferSize());
        }
        module_file_system->hide(binary);
    }
}

std::optional<std::filesystem::path>
SlangSession::canonical_file_path(const std::filesystem::path& path) const {
    const std::optional<std::filesystem::path> found =
        path.is_absolute() ? std::optional{path}
                           : compile_context()->get_search_path_file_loader().find_file(path);
    std::error_code ec;
    if (!found || !std::filesystem::is_regular_file(*found, ec)) {
        return std::nullopt;
    }
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(*found, ec);
    return ec ? *found : canonical;
}

std::optional<std::string>
SlangSession::canonical_module_name(const std::filesystem::path& file) const {
    const std::vector<std::filesystem::path>& search_paths =
        compile_context()->get_search_path_file_loader().get_search_paths();
    std::vector<std::filesystem::path> relative_paths;
    for (const std::filesystem::path& search_path : search_paths) {
        std::error_code ec;
        const std::filesystem::path relative =
            file.lexically_relative(std::filesystem::weakly_canonical(search_path, ec));
        if (!ec && !relative.empty() && *relative.begin() != "..") {
            relative_paths.push_back(relative);
        }
    }
    std::ranges::stable_sort(relative_paths, {}, [](const std::filesystem::path& relative) {
        return std::distance(relative.begin(), relative.end());
    });

    for (const std::filesystem::path& relative : relative_paths) {
        const std::string name = module_name_of_import(slang_import_spelling(relative));
        if (resolve_module(name, search_paths) == file) {
            return name;
        }
    }
    return std::nullopt;
}

bool SlangSession::modules_canonically_named() {
    if (!module_names_canonical) {
        return false;
    }
    for (SlangInt i = 0; i < session->getLoadedModuleCount(); i++) {
        slang::IModule* module = session->getLoadedModule(i);
        if (!checked_module_names.insert(module).second) {
            continue;
        }
        // a module loaded from a binary keeps the names it was compiled with
        const char* path = module->getFilePath();
        if (path == nullptr || std::filesystem::path{path}.extension() != ".slang" ||
            source_strings_by_identity.contains(path)) {
            continue;
        }
        const std::optional<std::filesystem::path> file = canonical_file_path(path);
        if (!file) {
            continue;
        }
        const std::string name = canonical_module_name(*file).value_or(file->generic_string());
        if (name != module->getName()) {
            SPDLOG_WARN("slang: {} is imported as {}, compiled modules are not cached in this "
                        "session; import it as {}",
                        file->string(), import_of_module_name(module->getName()),
                        import_of_module_name(name));
            module_names_canonical = false;
            return false;
        }
    }
    return true;
}

std::filesystem::path SlangSession::module_cache_dir(const ShaderCompileContext& context) {
    std::size_t seed = 0;
    for (const auto& [key, value] : context.get_preprocessor_macros()) {
        hash_combine(seed, key, value);
    }
    // module names and the paths Slang records depend on the search paths
    for (const std::filesystem::path& search_path : context.get_search_path_file_loader()) {
        std::error_code ec;
        hash_combine(seed, std::filesystem::weakly_canonical(search_path, ec).string());
    }
    hash_combine(seed, static_cast<uint32_t>(context.get_target()),
                 context.get_optimization_level(),
                 static_cast<uint32_t>(debug_info_level(context.should_generate_debug_info())));
    return cache_dir("slang-modules") / fmt::format("{:016x}", seed);
}

Slang::ComPtr<ISlangFileSystemExt>
SlangSession::create_file_system(const std::filesystem::path& module_cache_dir,
                                 std::function<bool()> serve_binaries) {
    return Slang::ComPtr<ISlangFileSystemExt>(
        new ModuleFileSystem(module_cache_dir, std::move(serve_binaries)));
}

bool SlangSession::is_imported(slang::IModule* module) const {
    const char* identity = module->getFilePath();
    if (identity == nullptr) {
        return true;
    }
    for (SlangInt i = 0; i < session->getLoadedModuleCount(); i++) {
        slang::IModule* loaded = session->getLoadedModule(i);
        if (loaded == module) {
            continue;
        }
        for (SlangInt32 j = 0; j < loaded->getDependencyFileCount(); j++) {
            if (std::string_view{loaded->getDependencyFilePath(j)} == identity) {
                return true;
            }
        }
    }
    return false;
}

bool SlangSession::is_replaced(slang::IModule* module) const {
    for (const auto& [name_and_hash, loaded] : module_versions) {
        if (loaded.get() == module) {
            return bound_modules.at(name_and_hash.first).module.get() != module;
        }
    }
    return false;
}

void SlangSession::check_no_replaced_dependency(slang::IModule* module) const {
    for (SlangInt32 i = 0; i < module->getDependencyFileCount(); i++) {
        const auto it = source_strings_by_identity.find(module->getDependencyFilePath(i));
        if (it != source_strings_by_identity.end() && it->second != module &&
            is_replaced(it->second)) {
            throw stale_session(
                fmt::format("module {} imports a replaced module source", module->getName()));
        }
    }
}

const Slang::ComPtr<slang::IModule>&
SlangSession::ensure_module(SlangComposition::SlangModule& module) {
    const std::string& name = module.get_name();
    const uint64_t source_hash = module.source_hash();

    const auto it = bound_modules.find(name);
    if (it != bound_modules.end()) {
        if (it->second.source_hash == source_hash) {
            return it->second.module;
        }
        if (is_imported(it->second.module)) {
            throw stale_session(
                fmt::format("module {} is rebound to a different source while imported", name));
        }
        if (module_versions.size() - bound_modules.size() >= MAX_REPLACED_MODULES) {
            throw stale_session("too many replaced module sources");
        }
        std::erase_if(entry_point_cache,
                      [&](const auto& entry) { return entry.first.get_module() == name; });
        type_conformance_cache.clear();
        composition_cache.clear();
    }

    // a source can be loaded only once per session
    auto [version, inserted] = module_versions.try_emplace(std::pair{name, source_hash});
    if (inserted) {
        const std::string slang_name =
            it == bound_modules.end() ? name : fmt::format("{}_{:016x}", name, source_hash);
        try {
            version->second = load_module_from_source(
                slang_name, module.get_source(compile_context()->get_search_path_file_loader()),
                module.get_import_path());
        } catch (...) {
            module_versions.erase(version);
            throw;
        }
    }
    return bound_modules.insert_or_assign(name, BoundModule{source_hash, version->second})
        .first->second.module;
}

Slang::ComPtr<slang::IComponentType>
SlangSession::compose(const SlangCompositionHandle& composition) {
    std::unordered_map<std::string, uint64_t> sources;
    check_one_source_per_name(composition, sources);
    const Slang::ComPtr<slang::IComponentType> composed = compose_tree(composition);
    store_modules();
    return composed;
}

void SlangSession::check_one_source_per_name(const SlangCompositionHandle& composition,
                                             std::unordered_map<std::string, uint64_t>& sources) {
    for (const auto& subcomposition : composition->compositions) {
        check_one_source_per_name(subcomposition, sources);
    }
    for (auto& module : composition->modules) {
        const auto [it, inserted] = sources.try_emplace(module.get_name(), module.source_hash());
        if (!inserted && it->second != module.source_hash()) {
            throw ShaderCompiler::compilation_failed(
                fmt::format("module {} is bound to two sources in one composition; give the "
                            "modules distinct names",
                            module.get_name()));
        }
    }
}

Slang::ComPtr<slang::IComponentType>
SlangSession::compose_tree(const SlangCompositionHandle& composition) {
    // owning, a rebind below may drop the cached composites
    std::vector<Slang::ComPtr<slang::IComponentType>> components;
    components.reserve(
        std::max(composition->modules.size() + composition->compositions.size(),
                 1 + composition->type_conformances.size() + composition->entry_points.size()));

    for (const auto& subcomposition : composition->compositions) {
        const uint64_t version = subcomposition->version();
        auto it = composition_cache.find(subcomposition);
        if (it == composition_cache.end() || it->second.first != version) {
            const Slang::ComPtr<slang::IComponentType> composed = compose_tree(subcomposition);
            it = composition_cache.insert_or_assign(subcomposition, std::pair{version, composed})
                     .first;
        }
        components.emplace_back(it->second.second);
    }

    std::set<SlangComposition::EntryPoint> additional_entry_points;
    for (auto& module : composition->modules) {
        Slang::ComPtr<slang::IModule> slang_module = ensure_module(module);
        components.emplace_back(slang_module.get());

        if (module.get_with_entry_points()) {
            for (uint32_t entry_point_index = 0;
                 entry_point_index < get_defined_entry_point_count(slang_module);
                 entry_point_index++) {
                Slang::ComPtr<slang::IEntryPoint> entry_point =
                    get_defined_entry_point(slang_module, entry_point_index);
                const char* name = entry_point->getFunctionReflection()->getName();
                auto rename_it = module.get_entry_point_map().find(name);
                if (rename_it == module.get_entry_point_map().end()) {
                    additional_entry_points.insert(
                        SlangComposition::EntryPoint(name, module.get_name()));
                } else {
                    additional_entry_points.insert(
                        SlangComposition::EntryPoint(name, module.get_name(), rename_it->second));
                }
            }
        }
    }

    Slang::ComPtr<slang::IComponentType> composed_modules = compose(components);
    components.clear();
    components.emplace_back(composed_modules);

    for (auto& [type_conformance, c_id] : composition->type_conformances) {
        auto it = type_conformance_cache.find(type_conformance);
        if (it == type_conformance_cache.end()) {
            int64_t id = c_id;
            it = type_conformance_cache
                     .emplace(type_conformance,
                              create_type_conformance(composed_modules,
                                                      type_conformance.get_type_name(),
                                                      type_conformance.get_interface_name(), id))
                     .first;
        }
        components.emplace_back(it->second);
    }

    for (const auto& ep : std::views::join(std::array{std::views::all(composition->entry_points),
                                                      std::views::all(additional_entry_points)})) {
        auto it = entry_point_cache.find(ep);
        if (it == entry_point_cache.end()) {
            Slang::ComPtr<slang::IModule> module = bound_modules.at(ep.get_module()).module;

            it = entry_point_cache
                     .emplace(
                         ep, std::make_pair(find_entry_point_or_fail(module, ep.get_defined_name()),
                                            nullptr))
                     .first;

            it->second.first->renameEntryPoint(ep.get_export_name().c_str(),
                                               it->second.second.writeRef());
        }
        components.push_back(it->second.second);
    }

    return compose(components);
}

Slang::ComPtr<slang::IBlob>
SlangSession::compile(const Slang::ComPtr<slang::IComponentType>& linked_programm,
                      const uint32_t entrypoint_index) {
    const auto start = std::chrono::steady_clock::now();
    const char* name =
        linked_programm->getLayout()->getEntryPointByIndex(entrypoint_index)->getNameOverride();

    const std::optional<std::filesystem::path> spv_path =
        spirv_cache_path(linked_programm, entrypoint_index);
    if (spv_path) {
        if (Slang::ComPtr<slang::IBlob> cached = cache_read(*spv_path)) {
            SPDLOG_INFO("slang: entry point {} loaded from cache ({:.1f} ms)", name,
                        milliseconds_since(start));
            return cached;
        }
    }

    Slang::ComPtr<slang::IBlob> compiled;
    Slang::ComPtr<slang::IBlob> diagnostics_blob;

    SlangResult result =
        linked_programm->getEntryPointCode(entrypoint_index,
                                           0, // targetIndex, currently only one supported
                                           compiled.writeRef(), diagnostics_blob.writeRef());

    if (SLANG_FAILED(result)) {
        throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
    }

    if (diagnostics_blob != nullptr) {
        SPDLOG_DEBUG("Slang compiling. Diagnostics: {}", diagnostics_as_string(diagnostics_blob));
    }

    if (spv_path) {
        cache_write(*spv_path, compiled->getBufferPointer(), compiled->getBufferSize());
    }
    SPDLOG_INFO("slang: entry point {} compiled ({:.1f} ms)", name, milliseconds_since(start));

    return compiled;
}

Slang::ComPtr<slang::IBlob>
SlangSession::compile(const Slang::ComPtr<slang::IComponentType>& linked_programm) {
    const auto start = std::chrono::steady_clock::now();

    const std::optional<std::filesystem::path> spv_path = spirv_cache_path(linked_programm);
    if (spv_path) {
        if (Slang::ComPtr<slang::IBlob> cached = cache_read(*spv_path)) {
            SPDLOG_INFO("slang: entry points {} loaded from cache ({:.1f} ms)",
                        entry_point_names(linked_programm), milliseconds_since(start));
            return cached;
        }
    }

    Slang::ComPtr<slang::IBlob> compiled;
    Slang::ComPtr<slang::IBlob> diagnostics_blob;

    SlangResult result =
        linked_programm->getTargetCode(0, // targetIndex, currently only one supported,
                                       compiled.writeRef(), diagnostics_blob.writeRef());

    if (SLANG_FAILED(result)) {
        throw ShaderCompiler::compilation_failed(diagnostics_as_string(diagnostics_blob));
    }

    if (diagnostics_blob != nullptr) {
        SPDLOG_DEBUG("Slang compiling. Diagnostics: {}", diagnostics_as_string(diagnostics_blob));
    }

    if (spv_path) {
        cache_write(*spv_path, compiled->getBufferPointer(), compiled->getBufferSize());
    }
    SPDLOG_INFO("slang: entry points {} compiled ({:.1f} ms)", entry_point_names(linked_programm),
                milliseconds_since(start));

    return compiled;
}

std::optional<std::filesystem::path>
SlangSession::spirv_cache_path(const Slang::ComPtr<slang::IComponentType>& program) {
    if (!cache_enabled()) {
        return std::nullopt;
    }
    const uint32_t count = static_cast<uint32_t>(program->getLayout()->getEntryPointCount());
    if (count == 0) {
        return std::nullopt;
    }
    std::string entry_point_hashes;
    for (uint32_t i = 0; i < count; i++) {
        Slang::ComPtr<slang::IBlob> hash;
        program->getEntryPointHash(static_cast<SlangInt>(i), 0, hash.writeRef());
        if (hash == nullptr || hash->getBufferSize() == 0) {
            return std::nullopt;
        }
        entry_point_hashes += to_hex(hash->getBufferPointer(), hash->getBufferSize());
    }
    // the concatenated entry point hashes exceed MAX_PATH on Windows
    return cache_dir("spirv") / fmt::format("{:016x}-{}.spv", hash_val(entry_point_hashes), count);
}

std::optional<std::filesystem::path>
SlangSession::spirv_cache_path(const Slang::ComPtr<slang::IComponentType>& program,
                               const uint32_t entry_point_index) {
    if (!cache_enabled()) {
        return std::nullopt;
    }
    Slang::ComPtr<slang::IBlob> hash;
    program->getEntryPointHash(static_cast<SlangInt>(entry_point_index), 0, hash.writeRef());
    if (hash == nullptr || hash->getBufferSize() == 0) {
        return std::nullopt;
    }
    return cache_dir("spirv") / (to_hex(hash->getBufferPointer(), hash->getBufferSize()) + ".spv");
}

SlangSessionHandle ShaderCompileContext::current_session() {
    const uint64_t epoch = slang_source_epoch();
    if (!hot_session || hot_session_epoch != epoch) {
        hot_session = SlangSession::create(shared_from_this());
        hot_session_epoch = epoch;
    }
    return hot_session;
}

static slang::TypeLayoutReflection* find_type_layout(slang::ProgramLayout* layout,
                                                     const std::string& type_name) {
    slang::TypeReflection* type = layout->findTypeByName(type_name.c_str());
    if (type == nullptr) {
        throw ShaderCompiler::compilation_failed(fmt::format("type '{}' not found", type_name));
    }

    slang::TypeLayoutReflection* type_layout =
        layout->getTypeLayout(type, slang::LayoutRules::Default);
    if (type_layout == nullptr) {
        throw ShaderCompiler::compilation_failed(
            fmt::format("failed to get type layout for '{}'", type_name));
    }

    return type_layout;
}

SlangSession::TypeLayoutResult
SlangSession::get_type_layout(const ShaderCompileContextHandle& compile_context,
                              const SlangCompositionHandle& composition,
                              const std::string& type_name) {
    const SlangProgramHandle program = SlangProgram::create(compile_context, composition).get();
    auto* type_layout = find_type_layout(program->get_program_reflection(), type_name);
    return {type_layout, program};
}

SlangSession::TypeLayoutResult
SlangSession::get_type_layout(const ShaderCompileContextHandle& compile_context,
                              const std::filesystem::path& module_path,
                              const std::string& type_name) {
    const SlangProgramHandle program =
        SlangProgram::create(compile_context, module_path, false).get();
    auto* type_layout = find_type_layout(program->get_program_reflection(), type_name);
    return {type_layout, program};
}

} // namespace merian
