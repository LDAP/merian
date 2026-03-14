#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionSDL : public ContextExtension {
  public:
    ExtensionSDL();

    ~ExtensionSDL();

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

  private:
    bool sdl_initialized = false;
};

} // namespace merian
