#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/utils/hash.hpp"
#include "merian/vk/shader/shader_compiler.hpp"

#include "slang-com-ptr.h"
#include "slang.h"

#include <map>
#include <memory>
#include <optional>
#include <string>

namespace merian {

class SlangComposition;
using SlangCompositionHandle = std::shared_ptr<SlangComposition>;

// Describes a composition of modules, type conformances, and entrypoints under a compile context.
// Which can be (lazyly) compiled to a SlangProgram and entry points.
//
// A slang composition used a single slang session for compilation.
class SlangComposition : public std::enable_shared_from_this<SlangComposition> {
    friend class SlangSession;

  private:
    SlangComposition();

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

        const bool& get_with_entry_points() const {
            return with_entry_points;
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
        static SlangModule
        from_path(const std::filesystem::path& path,
                  const bool with_entry_points,
                  const std::map<std::string, std::string>& entry_point_renames = {}) {
            SlangModule sm;

            sm.name = path.stem().string();
            sm.import_path = path.string();
            sm.source_path = path;
            sm.with_entry_points = with_entry_points;
            sm.entry_points_map = entry_point_renames;

            return sm;
        }

        static SlangModule
        from_source(const std::string& name,
                    const std::string& source,
                    const std::optional<std::string>& import_path,
                    const bool with_entry_points,
                    const std::map<std::string, std::string>& entry_point_renames = {}) {
            SlangModule sm;

            sm.name = name;
            sm.source = source;
            if (import_path) {
                sm.import_path = import_path;
            }
            sm.with_entry_points = with_entry_points;
            sm.entry_points_map = entry_point_renames;

            return sm;
        }

      private:
        std::string name;
        std::optional<std::string> import_path;

        std::optional<std::string> source;
        std::optional<std::filesystem::path> source_path{};

        bool with_entry_points;

        // name in module, exported name in composite
        std::map<std::string, std::string> entry_points_map;

        Slang::ComPtr<slang::IModule> module;
    };

    class TypeConformance {

      public:
        TypeConformance(const std::string& interface_name, const std::string& type_name)
            : interface_name(interface_name), type_name(type_name) {}

        const std::string& get_interface_name() const {
            return interface_name;
        }
        const std::string& get_type_name() const {
            return type_name;
        }

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

      public:
        EntryPoint(const std::string& defined_name, const std::string& from_module)
            : defined_name(defined_name), module(from_module), export_name(defined_name) {}

        EntryPoint(const std::string& defined_name,
                   const std::string& from_module,
                   const std::string& export_name)
            : defined_name(defined_name), module(from_module), export_name(export_name) {}

        const std::string& get_defined_name() const {
            return defined_name;
        }

        const std::string& get_module() const {
            return module;
        }

        const std::string& get_export_name() const {
            return export_name;
        }

        bool operator<(const EntryPoint& other) const {
            return module < other.module ||
                   (module == other.module && defined_name < other.defined_name) ||
                   (module == other.module && defined_name == other.defined_name &&
                    export_name < other.export_name);
        }

        bool operator==(const EntryPoint& other) const {
            return module == other.module && defined_name == other.defined_name &&
                   export_name == other.export_name;
        }

        struct HashFunction {
            size_t operator()(const EntryPoint& ep) const {
                return hash_val(ep.module, ep.defined_name, ep.export_name);
            }
        };

      private:
        std::string defined_name;
        std::string module;

        std::string export_name;
    };

    void add_module(const SlangModule& module);

    void add_module(SlangModule&& module);

    // shortcut for SlangModule::from_path
    void add_module_from_path(const std::filesystem::path& path,
                              const bool with_entry_points = false,
                              const std::map<std::string, std::string>& entry_point_renames = {});

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

    void add_composition(const SlangCompositionHandle& composition);

  public:
    static SlangCompositionHandle create();

  private:
    std::map<std::string, SlangModule> modules;
    // interface name -> type name -> dynamic dispach id
    std::map<TypeConformance, int64_t> type_conformances;
    std::set<EntryPoint> entry_points;
    std::set<SlangCompositionHandle> compositions;
};

} // namespace merian
