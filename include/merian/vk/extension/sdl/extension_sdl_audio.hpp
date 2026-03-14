#pragma once

#include "merian/vk/extension/extension.hpp"

#include <memory>

namespace merian {

class SDLAudioDevice;
using SDLAudioDeviceHandle = std::shared_ptr<SDLAudioDevice>;

class ExtensionSDLAudio : public ContextExtension {
  public:
    ExtensionSDLAudio();

    ~ExtensionSDLAudio();

    std::vector<std::string> request_extensions() override;

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

    SDLAudioDeviceHandle create_audio_device() const;

  private:
    bool audio_initialized = false;
};

} // namespace merian
