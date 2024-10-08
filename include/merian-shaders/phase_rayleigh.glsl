#ifndef _MERIAN_SHADERS_PHASE_RAYLEIGH_H_
#define _MERIAN_SHADERS_PHASE_RAYLEIGH_H_

float phase_rayleigh(const float cos_theta) {
    return 3 * (1 + cos_theta * cos_theta) / 4;
}

// Compute the amount of rayleigh for a certain wavelength
#define rayleigh_scattering(wavelength) (1 / pow(wavelength, 4))

// normalized scattering amount for wavelengths 700, 530, 440
#define rayleigh_sun vec3(0.13962466, 0.42486486, 0.89442429)

#endif
