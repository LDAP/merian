#include "merian/utils/audio/sdl_audio_device.hpp"

#include "SDL.h"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace merian {

static uint32_t audio_devices = 0;
static bool sdl_is_initialized = false;

static bool initialize_sdl() {
    if (sdl_is_initialized) {
        return true;
    }

    SPDLOG_DEBUG("initialize SDL");
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        return false;
    }

    sdl_is_initialized = true;
    return true;
}

static void uninitialize_sdl() {
    if (!sdl_is_initialized)
        return;

    if (audio_devices == 0) {
        SPDLOG_DEBUG("shutdown SDL");
        SDL_Quit();
        sdl_is_initialized = false;
    }
}

SDL_AudioFormat sdl_format(SDLAudioDevice::AudioFormat format) {
    switch (format) {
    case SDLAudioDevice::FORMAT_F32_LSB:
        return AUDIO_F32LSB;
    case SDLAudioDevice::FORMAT_S16_LSB:
        return AUDIO_S16LSB;
    default:
        throw std::runtime_error{"format unsupported"};
    }
}

SDLAudioDevice::AudioFormat merian_format(SDL_AudioFormat format) {
    switch (format) {
    case AUDIO_F32LSB:
        return SDLAudioDevice::FORMAT_F32_LSB;
    case AUDIO_S16LSB:
        return SDLAudioDevice::FORMAT_S16_LSB;
    default:
        throw std::runtime_error{"format unsupported"};
    }
}

static void SDLCALL sdl_callback(void* raw_callback, Uint8* stream, int len) {
    std::function<void(uint8_t * stream, int len)>* callback =
        static_cast<std::function<void(uint8_t * stream, int len)>*>(raw_callback);
    (*callback)(stream, len);
}

SDLAudioDevice::SDLAudioDevice() : AudioDevice() {

    audio_devices++;

    if (!initialize_sdl()) {
        SPDLOG_WARN("{}, disabling audio", SDL_GetError());
        audio_device_id = 0;
        return;
    }
}

SDLAudioDevice::~SDLAudioDevice() {
    audio_devices--;

    close_device();

    uninitialize_sdl();
}

std::optional<SDLAudioDevice::AudioSpec>
SDLAudioDevice::open_device(const AudioSpec& desired_audio_spec,
                            const std::function<void(uint8_t* stream, int len)>& callback,
                            const AllowedChangesFlags& allowed_changes) {
    this->callback = callback;

    int sdl_allowed_changes = 0;
    sdl_allowed_changes |= allowed_changes & AllowedChangesFlagBits::CHANNELS_CHANGE
                               ? SDL_AUDIO_ALLOW_CHANNELS_CHANGE
                               : 0;
    sdl_allowed_changes |= allowed_changes & AllowedChangesFlagBits::SAMPLERATE_CHANGE
                               ? SDL_AUDIO_ALLOW_FREQUENCY_CHANGE
                               : 0;
    sdl_allowed_changes |=
        allowed_changes & AllowedChangesFlagBits::FORMAT_CHANGE ? SDL_AUDIO_ALLOW_FORMAT_CHANGE : 0;
    sdl_allowed_changes |= allowed_changes & AllowedChangesFlagBits::BUFFERSIZE_CHANGE
                               ? SDL_AUDIO_ALLOW_SAMPLES_CHANGE
                               : 0;

    SDL_AudioSpec wanted_spec{
        desired_audio_spec.samplerate,
        sdl_format(desired_audio_spec.format),
        desired_audio_spec.channels,
        0,
        desired_audio_spec.buffersize,
        0,
        0,
        sdl_callback,
        callback ? &this->callback : nullptr,
    };
    SDL_AudioSpec audio_spec;
    audio_device_id = SDL_OpenAudioDevice(NULL, 0, &wanted_spec, &audio_spec, sdl_allowed_changes);
    if (audio_device_id) {
        this->audio_spec = AudioSpec{
            merian_format(audio_spec.format),
            audio_spec.samples,
            audio_spec.freq,
            audio_spec.channels,
        };

        SPDLOG_DEBUG("SDL audio device opened: {} Hz, {} samples, {} channels",
                     this->audio_spec->samplerate, this->audio_spec->buffersize,
                     this->audio_spec->channels);
    }

    return this->audio_spec;
}

void SDLAudioDevice::close_device() {
    if (audio_device_id) {
        SDL_CloseAudioDevice(audio_device_id);
        audio_device_id = 0;
        audio_spec.reset();
    }
}

// returns a audio spec if the device is open
std::optional<SDLAudioDevice::AudioSpec> SDLAudioDevice::get_audio_spec() {
    return audio_spec;
}

void SDLAudioDevice::queue_audio(const void* data, uint32_t len) {
    assert(!callback);

    if (audio_device_id) {
        SDL_QueueAudio(audio_device_id, data, len);
    }
}

void SDLAudioDevice::lock_device() {
    if (audio_device_id) {
        SDL_LockAudioDevice(audio_device_id);
    }
}

void SDLAudioDevice::unlock_device() {
    if (audio_device_id) {
        SDL_UnlockAudioDevice(audio_device_id);
    }
}

void SDLAudioDevice::unpause_audio() {
    if (audio_device_id) {
        SDL_PauseAudioDevice(audio_device_id, 0);
    }
}

void SDLAudioDevice::pause_audio() {
    if (audio_device_id) {
        SDL_PauseAudioDevice(audio_device_id, 1);
    }
}

} // namespace merian
