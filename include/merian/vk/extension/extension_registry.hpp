#pragma once

#include "merian/utils/pointer.hpp"
#include "merian/vk/extension/extension.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace merian {

template <typename ExtensionClass> std::shared_ptr<ContextExtension> create_extension() {
    return std::make_shared<ExtensionClass>();
}

// Type-erased priority entry for one provider interface. Constructed via ProviderPriority<T>.
struct ProviderPriorityEntry {
    std::type_index interface_type;
    int priority;
#ifndef NDEBUG
    std::function<bool(const std::shared_ptr<ContextExtension>&)> check_fn;
#endif
};

// Typed helper — construct as ProviderPriority<MyInterface>{50}. Implicitly converts to
// ProviderPriorityEntry. In debug builds the entry carries a runtime check that verifies the
// extension actually implements the interface.
template <typename InterfaceType> struct ProviderPriority {
    int priority;

    operator ProviderPriorityEntry() const {
        return {typeid(InterfaceType), priority,
#ifndef NDEBUG
                [](const std::shared_ptr<ContextExtension>& ext) {
                    return std::dynamic_pointer_cast<InterfaceType>(ext) != nullptr;
                }
#endif
        };
    }
};

class ExtensionRegistry {
  public:
    using ExtensionFactory = std::function<std::shared_ptr<ContextExtension>()>;

    ExtensionRegistry& operator=(const ExtensionRegistry&) = delete;
    ExtensionRegistry(const ExtensionRegistry&) = delete;

    // -----------------------------------

    static ExtensionRegistry& get_instance();

    // -----------------------------------

    template <typename EXTENSION_TYPE>
    void register_extension(const std::string& name,
                            const ExtensionFactory& factory,
                            const bool auto_load = false,
                            std::initializer_list<ProviderPriorityEntry> provider_priorities = {}) {
        const std::type_index type = typeid(std::remove_pointer_t<EXTENSION_TYPE>);

        if (type_to_name.contains(type)) {
            throw std::invalid_argument{
                fmt::format("extension with type {} already exists.", type.name())};
        }
        if (name_to_factory.contains(name)) {
            throw std::invalid_argument{
                fmt::format("extension with name {} already exists.", name)};
        }

        type_to_name[type] = name;
        name_to_factory[name] = factory;
        for (const auto& e : provider_priorities)
            name_to_provider_priority[name][e.interface_type] = e.priority;
        if (auto_load)
            auto_load_extension_names.push_back(name);

#ifndef NDEBUG
        if (!provider_priorities.size())
            return;
        auto test_instance = factory();
        for (const auto& e : provider_priorities) {
            if (!e.check_fn(test_instance)) {
                throw std::invalid_argument{
                    fmt::format("extension '{}' does not implement provider '{}'", name,
                                e.interface_type.name())};
            }
        }
#endif
    }

    template <typename EXTENSION_TYPE>
    void register_extension(const std::string& name,
                            const bool auto_load = false,
                            std::initializer_list<ProviderPriorityEntry> provider_priorities = {}) {
        register_extension<EXTENSION_TYPE>(name, create_extension<EXTENSION_TYPE>, auto_load,
                                           provider_priorities);
    }

    // Names of extensions registered with auto_load=true, in registration order.
    const std::vector<std::string>& get_auto_load_names() const {
        return auto_load_extension_names;
    }

    std::shared_ptr<ContextExtension> create(const std::string& name) const;

    bool is_registered(const std::string& name) const;

    template <typename EXTENSION_TYPE> bool is_registered() const {
        const std::type_index type = typeid(std::remove_pointer_t<EXTENSION_TYPE>);
        return type_to_name.contains(type);
    }

    template <typename EXTENSION_TYPE> const std::string& get_name() const {
        const std::type_index type = typeid(std::remove_pointer_t<EXTENSION_TYPE>);
        if (!type_to_name.contains(type)) {
            throw std::invalid_argument{
                fmt::format("extension with type {} is not registerd.", type.name())};
        }

        return type_to_name.at(type);
    }

    template <typename PTR_TYPE> const std::string& get_name(const PTR_TYPE extension) const {
        const std::type_index type = typeindex_from_pointer(extension);
        if (!type_to_name.contains(type)) {
            throw std::invalid_argument{
                fmt::format("extension with type {} is not registerd.", type.name())};
        }

        return type_to_name.at(type);
    }

    std::vector<std::string> get_registered_extensions() const;

    // Returns the registered priority for (name, interface_type), or 0 if not set.
    int get_priority(const std::string& name, const std::type_index& interface_type) const;

  private:
    ExtensionRegistry();

    std::unordered_map<std::type_index, std::string> type_to_name;
    std::unordered_map<std::string, ExtensionFactory> name_to_factory;
    std::unordered_map<std::string, std::unordered_map<std::type_index, int>>
        name_to_provider_priority;
    std::vector<std::string> auto_load_extension_names;
};

template <typename EXTENSION_TYPE> class ExtensionRegisterer {
  public:
    ExtensionRegisterer(const std::string& name) {
        auto& registry = ExtensionRegistry::get_instance();

        if (registry.is_registered<EXTENSION_TYPE>()) {
            return;
        }
        if (registry.is_registered(name)) {
            return;
        }

        registry.register_extension<EXTENSION_TYPE>(name);
    }
};

} // namespace merian

#define REGISTER_CONTEXT_EXTENSION(EXTENSION_TYPE, NAME)                                           \
    namespace {                                                                                    \
    merian::ExtensionRegisterer<EXTENSION_TYPE> register_ext_##EXTENSION_NAME(NAME);               \
    }
