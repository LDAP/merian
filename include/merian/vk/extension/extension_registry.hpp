#pragma once

#include "merian/vk/extension/extension.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace merian {

class ExtensionRegistry {
  public:
    using ExtensionFactory = std::function<std::shared_ptr<ContextExtension>()>;

    static ExtensionRegistry& get_instance();

    void register_extension(const std::string& name, const ExtensionFactory& factory);

    std::shared_ptr<ContextExtension> create(const std::string& name) const;

    bool is_registered(const std::string& name) const;

    std::vector<std::string> get_registered_extensions() const;

  private:
    ExtensionRegistry();
    std::unordered_map<std::string, ExtensionFactory> registry;
};

template <typename ExtensionClass>
std::shared_ptr<ContextExtension> create_extension() {
    return std::make_shared<ExtensionClass>();
}

} // namespace merian
