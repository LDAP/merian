#ifndef _MERIAN_SHADERS_COLORS_DNG_H_
#define _MERIAN_SHADERS_COLORS_DNG_H_

/*
Adapted from VKDT, licensed under:

copyright 2019 johannes hanika

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// apparently this is what the dng spec says should be used to correct colour
// after application of tone-changing operators (curve). with all the sorting,
// it's based on hsv and designed to keep hue constant in that space.
vec3 // returns hue mapped rgb colour
adjust_colour_dng(
    vec3 col0, // original colour
    vec3 col1) // mapped colour, for instance by rgb curves
{
  bvec3 flip = bvec3(false); // bubble sort
  if(col0.b > col0.g) { col0.bg = col0.gb; col1.bg = col1.gb; flip.x = true; }
  if(col0.g > col0.r) { col0.rg = col0.gr; col1.rg = col1.gr; flip.y = true; }
  if(col0.b > col0.g) { col0.bg = col0.gb; col1.bg = col1.gb; flip.z = true; }

  col1.g = mix(col1.b, col1.r, (col0.g-col0.b+1e-8)/(col0.r-col0.b+1e-8));

  if(flip.z) col1.bg = col1.gb;
  if(flip.y) col1.rg = col1.gr;
  if(flip.x) col1.bg = col1.gb;
  return col1;
}

#endif
