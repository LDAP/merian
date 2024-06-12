#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <memory>

namespace merian {

class AudioDevice : public std::enable_shared_from_this<AudioDevice> {
  public:
    // Convention lower 0xFF are bit count (following what SDL does).
    enum AudioFormat {
        FORMAT_S16_LSB = 0x8010,
        FORMAT_F32_LSB = 0x8020,
        FORMAT_U8 = 0x0008,
        FORMAT_S8 = 0x8008,
        FORMAT_U16LSB = 0x0010,
        FORMAT_S16LSB = 0x8010,
        FORMAT_U16MSB = 0x1010,
        FORMAT_S16MSB = 0x9010,
    };

    struct AudioSpec {
        AudioFormat format;
        uint16_t buffersize;
        int samplerate;
        unsigned char channels;
    };

    AudioDevice() {}

    virtual ~AudioDevice() = 0;

    // Open a audio device with the desired format specification.
    // This method returns the obtained AudioSpec, if it succeeds.
    virtual std::optional<AudioSpec>
    open_device(const AudioSpec& desired_audio_spec,
                const std::function<void(uint8_t* stream, int len)>& callback) = 0;

    virtual void close_device() = 0;

    // returns a audio spec if the device is open
    virtual std::optional<AudioSpec> get_audio_spec() = 0;

    // Pauses the audio device from calling the callback, to safely change the data.
    // Locking multiple times is possible, however you must call unlock_device the same number of
    // times!
    virtual void lock_device() = 0;

    virtual void unlock_device() = 0;

    // Pauses the audio device from calling the callback, e.g. to initialize variables.
    // While paused, silence is written to the audio device, meaning this is not suitable to just
    // change some variables as it will result in dropouts, use lock_device for that.
    virtual void pause_audio() = 0;

    virtual void unpause_audio() = 0;
};

} // namespace merian
