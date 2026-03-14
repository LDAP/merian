#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/sdl/extension_sdl.hpp"

#include <memory>

namespace merian {

class SDLAudioDevice;
using SDLAudioDeviceHandle = std::shared_ptr<SDLAudioDevice>;

class ExtensionSDLAudio : public ContextExtension {
  public:
    static constexpr const char* name = "SDL Audio";

    ExtensionSDLAudio();

    ~ExtensionSDLAudio();

    std::vector<std::string> request_extensions() override;

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

    SDLAudioDeviceHandle create_audio_device() const;

  private:
    std::shared_ptr<ExtensionSDL> sdl_ext;
    bool audio_initialized = false;
};

} // namespace merian
