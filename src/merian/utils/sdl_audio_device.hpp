#pragma once

#include <cstdint>
#include <functional>

namespace merian {

class SDLAudioDevice {
  public:
    enum AudioFormat { FORMAT_S16_LSB, FORMAT_F32_LSB };

    SDLAudioDevice(AudioFormat format,
                   std::function<void(uint8_t* stream, int len)> callback,
                   uint16_t buffersize = 1024,
                   int samplerate = 44100,
                   unsigned char channels = 2);

    ~SDLAudioDevice();

    void unpause_audio();

    void pause_audio();

  private:
    unsigned int audio_device_id;
    std::function<void(uint8_t* stream, int len)> callback;
};

} // namespace merian
