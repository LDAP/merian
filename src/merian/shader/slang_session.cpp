#include "merian/shader/slang_session.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/hash.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <system_error>
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

std::filesystem::path SlangSession::cache_dir(const std::string_view subdir) {
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
    return cache_root() / tag / subdir;
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

    uint64_t budget = 128ull * 1024 * 1024;
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

uint64_t SlangSession::ir_cache_key(const std::string& name,
                                    const std::string& source,
                                    const std::optional<std::filesystem::path>& path) const {
    std::size_t seed = 0;
    hash_combine(seed, name);
    hash_combine(seed, path ? path->string() : std::string{});
    hash_combine(seed, source);
    for (const auto& [key, value] : shader_compile_context->get_preprocessor_macros()) {
        hash_combine(seed, key, value);
    }
    hash_combine(seed, static_cast<uint32_t>(shader_compile_context->get_target()),
                 shader_compile_context->get_optimization_level(),
                 static_cast<uint32_t>(shader_compile_context->should_generate_debug_info()));
    return seed;
}

std::optional<std::filesystem::path>
SlangSession::spirv_cache_path(const Slang::ComPtr<slang::IComponentType>& program) {
    if (!cache_enabled()) {
        return std::nullopt;
    }
    slang::ProgramLayout* layout = program->getLayout();
    if (layout == nullptr) {
        return std::nullopt;
    }
    const uint32_t count = static_cast<uint32_t>(layout->getEntryPointCount());
    if (count == 0) {
        return std::nullopt;
    }
    std::string name;
    for (uint32_t i = 0; i < count; i++) {
        Slang::ComPtr<slang::IBlob> hash;
        program->getEntryPointHash(static_cast<SlangInt>(i), 0, hash.writeRef());
        if (hash == nullptr || hash->getBufferSize() == 0) {
            return std::nullopt;
        }
        name += to_hex(hash->getBufferPointer(), hash->getBufferSize());
    }
    return cache_dir("spirv") / (name + ".spv");
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
