#include "merian/vk/extension/sdl/sdl_audio_device.hpp"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace merian {

static SDL_AudioFormat sdl_format(SDLAudioDevice::AudioFormat format) {
    switch (format) {
    case SDLAudioDevice::FORMAT_F32_LSB:
        return SDL_AUDIO_F32LE;
    case SDLAudioDevice::FORMAT_S16_LSB:
        return SDL_AUDIO_S16LE;
    default:
        throw std::runtime_error{"format unsupported"};
    }
}

static SDLAudioDevice::AudioFormat merian_format(SDL_AudioFormat format) {
    switch (format) {
    case SDL_AUDIO_F32LE:
        return SDLAudioDevice::FORMAT_F32_LSB;
    case SDL_AUDIO_S16LE:
        return SDLAudioDevice::FORMAT_S16_LSB;
    default:
        throw std::runtime_error{"format unsupported"};
    }
}

static void SDLCALL stream_callback(void* userdata,
                                    SDL_AudioStream* stream,
                                    int additional_amount,
                                    int /*total_amount*/) {
    auto* cb = static_cast<std::function<void(uint8_t*, int)>*>(userdata);
    if (!cb || !*cb)
        return;
    std::vector<uint8_t> buf(additional_amount);
    (*cb)(buf.data(), additional_amount);
    SDL_PutAudioStreamData(stream, buf.data(), additional_amount);
}

SDLAudioDevice::SDLAudioDevice() : AudioDevice() {}

SDLAudioDevice::~SDLAudioDevice() {
    close_device();
}

std::optional<SDLAudioDevice::AudioSpec>
SDLAudioDevice::open_device(const AudioSpec& desired_audio_spec,
                            const std::function<void(uint8_t* stream, int len)>& cb) {
    callback = cb;

    SDL_AudioSpec spec{};
    spec.format   = sdl_format(desired_audio_spec.format);
    spec.channels = desired_audio_spec.channels;
    spec.freq     = desired_audio_spec.samplerate;

    if (callback) {
        audio_stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec,
                                                 stream_callback, &callback);
    } else {
        audio_stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec,
                                                 nullptr, nullptr);
    }

    if (!audio_stream) {
        SPDLOG_WARN("SDL_OpenAudioDeviceStream failed: {}", SDL_GetError());
        return std::nullopt;
    }

    SDL_AudioSpec obtained{};
    if (SDL_GetAudioStreamFormat(audio_stream, &obtained, nullptr)) {
        audio_spec = AudioSpec{
            merian_format(obtained.format),
            desired_audio_spec.buffersize,
            obtained.freq,
            static_cast<unsigned char>(obtained.channels),
        };
        SPDLOG_DEBUG("SDL audio device opened: {} Hz, {} channels", audio_spec->samplerate,
                     audio_spec->channels);
    }

    return audio_spec;
}

void SDLAudioDevice::close_device() {
    if (audio_stream) {
        SDL_DestroyAudioStream(audio_stream);
        audio_stream = nullptr;
        audio_spec.reset();
    }
}

std::optional<SDLAudioDevice::AudioSpec> SDLAudioDevice::get_audio_spec() {
    return audio_spec;
}

void SDLAudioDevice::queue_audio(const void* data, uint32_t len) {
    assert(!callback);
    if (audio_stream) {
        SDL_PutAudioStreamData(audio_stream, data, static_cast<int>(len));
    }
}

void SDLAudioDevice::lock_device() {
    if (audio_stream) {
        SDL_LockAudioStream(audio_stream);
    }
}

void SDLAudioDevice::unlock_device() {
    if (audio_stream) {
        SDL_UnlockAudioStream(audio_stream);
    }
}

void SDLAudioDevice::unpause_audio() {
    if (audio_stream) {
        SDL_ResumeAudioStreamDevice(audio_stream);
    }
}

void SDLAudioDevice::pause_audio() {
    if (audio_stream) {
        SDL_PauseAudioStreamDevice(audio_stream);
    }
}

} // namespace merian
