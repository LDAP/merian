#ifndef _MERIAN_SHADERS_PHASE_DRAINE_H_
#define _MERIAN_SHADERS_PHASE_DRAINE_H_

/*
 * SPDX-FileCopyrightText: Copyright (c) <2023> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// [Jendersie and d'Eon 2023]
//   An Approximate Mie Scattering Function for Fog and Cloud Rendering
//   SIGGRAPH 2023 Talks
//   https://doi.org/10.1145/3587421.3595409

// EVAL and SAMPLE for the Draine (and therefore Cornette-Shanks) phase function
//   g = HG shape parameter
//   a = "alpha" shape parameter

// Warning: these functions don't special case isotropic scattering and can numerically fail for certain inputs

// eval:
//   u = dot(prev_dir, next_dir)
float phase_draine_eval(const float u, const float g, const float a)
{
    return ((1 - g*g)*(1 + a*u*u))/(4.*(1 + (a*(1 + 2*g*g))/3.) * M_PI * pow(1 + g*g - 2*g*u,1.5));
}

// sample: (sample an exact deflection cosine)
//   xi = a uniform random real in [0,1]
float phase_draine_sample(const float xi, const float g, const float a)
{
    const float g2 = g * g;
    const float g3 = g * g2;
    const float g4 = g2 * g2;
    const float g6 = g2 * g4;
    const float pgp1_2 = (1 + g2) * (1 + g2);
    const float T1 = (-1 + g2) * (4 * g2 + a * pgp1_2);
    const float T1a = -a + a * g4;
    const float T1a3 = T1a * T1a * T1a;
    const float T2 = -1296 * (-1 + g2) * (a - a * g2) * (T1a) * (4 * g2 + a * pgp1_2);
    const float T3 = 3 * g2 * (1 + g * (-1 + 2 * xi)) + a * (2 + g2 + g3 * (1 + 2 * g2) * (-1 + 2 * xi));
    const float T4a = 432 * T1a3 + T2 + 432 * (a - a * g2) * T3 * T3;
    const float T4b = -144 * a * g2 + 288 * a * g4 - 144 * a * g6;
    const float T4b3 = T4b * T4b * T4b;
    const float T4 = T4a + sqrt(-4 * T4b3 + T4a * T4a);
    const float T4p3 = pow(T4, 1.0 / 3.0);
    const float T6 = (2 * T1a + (48 * pow(2, 1.0 / 3.0) *
        (-(a * g2) + 2 * a * g4 - a * g6)) / T4p3 + T4p3 / (3. * pow(2, 1.0 / 3.0))) / (a - a * g2);
    const float T5 = 6 * (1 + g2) + T6;
    return (1 + g2 - pow(-0.5 * sqrt(T5) + sqrt(6 * (1 + g2) - (8 * T3) / (a * (-1 + g2) * sqrt(T5)) - T6) / 2., 2)) / (2. * g);
}

vec3 phase_draine_sample(const vec2 xi, const vec3 wi, const float g, const float a) {
    const float deflection_cos = phase_draine_sample(xi.x, g, a);
    const float z2 = sqrt(1.0 - deflection_cos * deflection_cos);
    return make_frame(wi) * vec3(z2 * cos(TWO_PI * xi.y), z2 * sin(TWO_PI * xi.y), deflection_cos);
}

#endif
