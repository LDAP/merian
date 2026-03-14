#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionSDL : public ContextExtension {
  public:
    static constexpr const char* name = "SDL";

    ExtensionSDL();

    ~ExtensionSDL();

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

  private:
    bool sdl_initialized = false;
};

} // namespace merian
