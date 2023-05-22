#pragma once

#include "vk/context.hpp"
#include <vector>

class Extension {
  public:
    virtual ~Extension() = 0;
    virtual std::string name() const = 0;
    std::vector<const char*> required_extension_names() const {
        return {};
    }
    std::vector<const char*> required_layer_names() const {
        return {};
    }
    void on_instance_created(vk::Instance&) {}
};
