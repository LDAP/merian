#include "sdl_audio_device.hpp"

#include "SDL.h"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace merian {

static uint32_t audio_devices = 0;

static void initialize_sdl() {
    static bool is_initialized = false;
    if (is_initialized)
        return;

    SPDLOG_DEBUG("initialize SDL");
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        throw std::runtime_error{SDL_GetError()};
    }

    audio_devices++;
    is_initialized = true;
}

static void uninitialize_sdl() {
    if (--audio_devices == 0) {
        SPDLOG_DEBUG("shutdown SDL");
        SDL_Quit();
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

static void SDLCALL sdl_callback(void* raw_callback, Uint8* stream, int len) {
    std::function<void(uint8_t * stream, int len)>* callback =
        static_cast<std::function<void(uint8_t * stream, int len)>*>(raw_callback);
    (*callback)(stream, len);
}

SDLAudioDevice::SDLAudioDevice(AudioFormat format,
                               std::function<void(uint8_t* stream, int len)> callback,
                               uint16_t buffersize,
                               int samplerate,
                               unsigned char channels)
    : callback(callback) {

    initialize_sdl();

    SDL_AudioSpec wanted_spec{samplerate, sdl_format(format), channels, 0, buffersize, 0,
                              0,          sdl_callback,       &this->callback};
    SDL_AudioSpec audio_spec;
    audio_device_id = SDL_OpenAudioDevice(NULL, 0, &wanted_spec, &audio_spec, 0);

    if (!audio_device_id) {
        throw std::runtime_error{SDL_GetError()};
    }
}

SDLAudioDevice::~SDLAudioDevice() {
    SDL_CloseAudioDevice(audio_device_id);
    uninitialize_sdl();
}

void SDLAudioDevice::unpause_audio() {
    SDL_PauseAudioDevice(audio_device_id, 0);
}

void SDLAudioDevice::pause_audio() {
    SDL_PauseAudioDevice(audio_device_id, 1);
}


} // namespace merian
