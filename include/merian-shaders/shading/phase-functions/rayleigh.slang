#pragma once

namespace merian {

float phase_rayleigh(float cos_theta) {
    return (3.0f * (1.0f + (cos_theta * cos_theta))) / 4.0f;
}

float rayleigh_scattering(float wavelength) {
    return 1.0f / pow(wavelength, 4.0f);
}

// normalized scattering amount for wavelengths 700, 530, 440
#define rayleigh_sun vec3(0.13962466, 0.42486486, 0.89442429)

}
