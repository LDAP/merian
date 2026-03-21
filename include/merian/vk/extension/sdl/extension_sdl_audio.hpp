#pragma once

#include "merian/utils/audio/audio_device_provider.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/sdl/extension_sdl.hpp"

#include <memory>

namespace merian {

class SDLAudioDevice;
using SDLAudioDeviceHandle = std::shared_ptr<SDLAudioDevice>;

class ExtensionSDLAudio : public ContextExtension, public AudioDeviceProvider {
  public:
    static constexpr const char* name = "SDL3 (Audio Subsystem)";

    ExtensionSDLAudio();

    ~ExtensionSDLAudio();

    std::vector<std::string> request_extensions() override;

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

    AudioDeviceHandle create_audio_device() const override;

  private:
    std::shared_ptr<ExtensionSDL> sdl_ext;
    bool audio_initialized = false;
};

} // namespace merian
