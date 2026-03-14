#pragma once

#include "merian/utils/audio/audio_device.hpp"

#include <cstdint>
#include <functional>
#include <optional>

// Forward declaration to avoid including SDL headers
struct SDL_AudioStream;

namespace merian {

class ExtensionSDLAudio;

class SDLAudioDevice : public AudioDevice {
  public:
    friend class ExtensionSDLAudio;

  private:
    SDLAudioDevice();

  public:
    ~SDLAudioDevice() override;

    std::optional<SDLAudioDevice::AudioSpec>
    open_device(const AudioSpec& desired_audio_spec,
                const std::function<void(uint8_t* stream, int len)>& callback = {}) override;

    void close_device() override;

    std::optional<AudioSpec> get_audio_spec() override;

    void queue_audio(const void* data, uint32_t len) override;

    void lock_device() override;

    void unlock_device() override;

    void pause_audio() override;

    void unpause_audio() override;

  private:
    SDL_AudioStream* audio_stream = nullptr;
    std::optional<AudioSpec> audio_spec;

    std::function<void(uint8_t* stream, int len)> callback;
};

} // namespace merian
