#pragma once

#include "merian/utils/audio/audio_device.hpp"

#include <cstdint>
#include <functional>
#include <optional>

namespace merian {

class SDLAudioDevice : public AudioDevice {
  public:
    SDLAudioDevice();

    ~SDLAudioDevice();

    // Open a audio device with the desired format specification.
    // This method returns the obtained AudioSpec, if it succeeds.
    std::optional<SDLAudioDevice::AudioSpec>
    open_device(const AudioSpec& desired_audio_spec,
                const std::function<void(uint8_t* stream, int len)>& callback) override;

    void close_device() override;

    // returns a audio spec if the device is open
    std::optional<AudioSpec> get_audio_spec() override;

    // Pauses the audio device from calling the callback, to safely change the data.
    // Locking multiple times is possible, however you must call unlock_device the same number of
    // times!
    void lock_device() override;

    void unlock_device() override;

    // Pauses the audio device from calling the callback, e.g. to initialize variables.
    // While paused, silence is written to the audio device, meaning this is not suitable to just
    // change some variables as it will result in dropouts, use lock_device for that.
    void pause_audio() override;

    void unpause_audio() override;

  private:
    unsigned int audio_device_id;
    std::optional<AudioSpec> audio_spec;

    std::function<void(uint8_t* stream, int len)> callback;
};

} // namespace merian
