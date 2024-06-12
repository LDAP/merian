#pragma once

#include <cstdint>
#include <functional>
#include <optional>

namespace merian {

class SDLAudioDevice {
  public:
    // Convention lower 0xFF are bit count.
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
        SDLAudioDevice::AudioFormat format;
        uint16_t buffersize;
        int samplerate;
        unsigned char channels;
    };

    SDLAudioDevice(std::function<void(uint8_t* stream, int len)> callback);

    ~SDLAudioDevice();

    // Open a audio device with the desired format specification.
    // This method returns the obtained AudioSpec, if it succeeds.
    std::optional<SDLAudioDevice::AudioSpec> open_device(const AudioSpec& desired_audio_spec);

    void close_device();

    // returns a audio spec if the device is open
    std::optional<AudioSpec> get_audio_spec();

    // Pauses the audio device from calling the callback, to safely change the data.
    // Locking multiple times is possible, however you must call unlock_device the same number of
    // times!
    void lock_device();

    void unlock_device();

    // Pauses the audio device from calling the callback, e.g. to initialize variables.
    // While paused, silence is written to the audio device, meaning this is not suitable to just
    // change some variables as it will result in dropouts, use lock_device for that.
    void pause_audio();

    void unpause_audio();

  private:
    unsigned int audio_device_id;
    std::optional<AudioSpec> audio_spec;

    std::function<void(uint8_t* stream, int len)> callback;
};

} // namespace merian
