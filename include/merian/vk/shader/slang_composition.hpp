#pragma once

#include "merian/utils/hash.hpp"
#include "merian/vk/shader/shader_compile_context.hpp"
#include "merian/vk/shader/slang_session.hpp"

namespace merian {

class SlangComposition;
using SlangCompositionHandle = std::shared_ptr<SlangComposition>;

// Describes a composition of modules, type conformances, and entrypoints under a compile context.
// Which can be (lazyly) compiled to a SlangProgram and entry points.
//
// A slang composition used a single slang session for compilation.
class SlangComposition {
  private:
    SlangComposition(const SlangSessionHandle& session);

  public:
    class SlangModule {
        friend class SlangComposition;

      private:
        SlangModule() {}

      public:
        const std::string& get_name() {
            return name;
        }
        // this is the path for path-based import
        const std::optional<std::string>& get_import_path() {
            return import_path;
        }

        // the entry points to include in the composite (name -> exported name)
        const std::map<std::string, std::string>& get_entry_point_map() {
            return entry_points_map;
        }

        void rename_entry_point(const std::string& name, const std::string& export_name) {
            entry_points_map[name] = export_name;
        }

        // file loader to resolve source path.
        const std::string& get_source(const FileLoader& file_loader) {
            if (source_path.has_value()) {
                source = file_loader.find_and_load_file(*source_path);
                if (!source) {
                    throw ShaderCompiler::compilation_failed{fmt::format(
                        "module source path {} could not be found", source_path->string())};
                }
            }

            assert(source.has_value());
            return *source;
        }

        // can be empty if is from source string.
        const std::optional<std::filesystem::path>& get_source_path() {
            return source_path;
        }

        // can be relative to search paths of the composits compile context.
        static SlangModule from_path(const std::filesystem::path& path) {
            SlangModule sm;

            sm.name = path.stem();
            sm.import_path = path.string();
            sm.source_path = path;

            return sm;
        }

        static SlangModule from_source(const std::string& name,
                                       const std::string& source,
                                       const std::optional<std::string>& import_path) {
            SlangModule sm;

            sm.name = name;
            sm.source = source;
            if (import_path) {
                sm.import_path = import_path;
            }

            return sm;
        }

      private:
        std::string name;
        std::optional<std::string> import_path;

        std::optional<std::string> source;
        std::optional<std::filesystem::path> source_path{};

        // name in module, exported name in composite
        std::map<std::string, std::string> entry_points_map;

        Slang::ComPtr<slang::IModule> module;
    };

    class TypeConformance {
        friend class SlangComposition;

      public:
        TypeConformance(const std::string& interface_name, const std::string& type_name)
            : interface_name(interface_name), type_name(type_name) {}

        bool operator<(const TypeConformance& other) const {
            return interface_name < other.interface_name ||
                   (interface_name == other.interface_name && type_name < other.type_name);
        }
        bool operator==(const TypeConformance& other) const {
            return type_name == other.type_name && interface_name == other.interface_name;
        }
        struct HashFunction {
            size_t operator()(const TypeConformance& conformance) const {
                return hash_val(conformance.type_name, conformance.interface_name);
            }
        };

      private:
        std::string interface_name;
        std::string type_name;
    };

    struct EntryPoint {
        friend class SlangComposition;

      private:
        EntryPoint(const std::string& defined_name, const std::string& from_module)
            : defined_name(defined_name), from_module(from_module) {}

      private:
        std::string defined_name;
        std::string from_module;

        Slang::ComPtr<slang::IEntryPoint> entry_point;
        Slang::ComPtr<slang::IComponentType> renamed_entry_point;
    };

    void add_module(const SlangModule& module, const bool with_entry_points);

    void add_module(SlangModule&& module, const bool with_entry_points);

    // shortcut for SlangModule::from_path
    void add_module_from_path(const std::filesystem::path& path, const bool with_entry_points);

    void add_type_conformance(const std::string& interface_name,
                              const std::string& type_name,
                              const int64_t dynamic_dispatch_id = -1);

    // can also update existing type conformance
    void add_type_conformance(const TypeConformance& type_conformance,
                              const int64_t dynamic_dispatch_id = -1);

    void add_type_conformance(TypeConformance&& type_conformance,
                              const int64_t dynamic_dispatch_id = -1);

    void add_entry_point(const std::string& defined_entry_point_name,
                         const std::string& from_module);

    Slang::ComPtr<slang::IComponentType> get_composite();

  public:
    static SlangCompositionHandle create(const SlangSessionHandle& session);

  private:
    void load_module(SlangModule& module);

    void add_entry_points_from_module(SlangModule& module);

    void compose();

  private:
    SlangSessionHandle session;

    std::map<std::string, SlangModule> modules;
    // interface name -> type name -> dynamic dispach id
    std::map<TypeConformance, std::pair<int64_t, Slang::ComPtr<slang::ITypeConformance>>>
        type_conformances;
    std::vector<EntryPoint> entry_points;

    // can be nullptr if something changed.
    Slang::ComPtr<slang::IComponentType> composite;
};

} // namespace merian
