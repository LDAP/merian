#pragma once

#include "merian/utils/audio/audio_device.hpp"

namespace merian {

// Interface for context extensions that can provide an AudioDevice.
// Use Context::find_provider<AudioDeviceProvider>() to locate one at runtime.
class AudioDeviceProvider {
  public:
    virtual ~AudioDeviceProvider() = default;

    virtual AudioDeviceHandle create_audio_device() const = 0;
};

} // namespace merian
