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

class ExtensionRegistry {
  public:
    using ExtensionFactory = std::function<std::shared_ptr<ContextExtension>()>;

    ExtensionRegistry& operator=(const ExtensionRegistry&) = delete;
    ExtensionRegistry(const ExtensionRegistry&) = delete;

    // -----------------------------------

    static ExtensionRegistry& get_instance();

    // -----------------------------------

    template <typename EXTENSION_TYPE>
    void register_extension(const std::string& name, const ExtensionFactory& factory) {
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
    }

    template <typename EXTENSION_TYPE> void register_extension(const std::string& name) {
        register_extension<EXTENSION_TYPE>(name, create_extension<EXTENSION_TYPE>);
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

  private:
    ExtensionRegistry();

    std::unordered_map<std::type_index, std::string> type_to_name;
    std::unordered_map<std::string, ExtensionFactory> name_to_factory;
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
