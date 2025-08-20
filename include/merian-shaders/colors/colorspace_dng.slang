#pragma once

implementing colorspaces;

namespace merian {

/*
Adapted from VKDT, licensed under:

copyright 2019 johannes hanika

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// apparently this is what the dng spec says should be used to correct colour
// after application of tone-changing operators (curve). with all the sorting,
// it's based on hsv and designed to keep hue constant in that space.
public float3 adjust_colour_dng(in float3 original, in float3 mapped) {
    bool3 flip = bool3(false, false, false);
    if (original.z > original.y) {
        float3 _36 = original;
        original.z = _36.yz.x;
        original.y = _36.yz.y;
        float3 _42 = mapped;
        mapped.z = _42.yz.x;
        mapped.y = _42.yz.y;
        flip.x = true;
    }
    if (original.y > original.x) {
        float3 _56 = original;
        original.x = _56.yx.x;
        original.y = _56.yx.y;
        float3 _62 = mapped;
        mapped.x = _62.yx.x;
        mapped.y = _62.yx.y;
        flip.y = true;
    }
    if (original.z > original.y) {
        float3 _76 = original;
        original.z = _76.yz.x;
        original.y = _76.yz.y;
        float3 _82 = mapped;
        mapped.z = _82.yz.x;
        mapped.y = _82.yz.y;
        flip.z = true;
    }
    mapped.y = lerp(mapped.z, mapped.x,
                    ((original.y - original.z) + 1e-8) / ((original.x - original.z) + 1e-8));
    if (flip.z) {
        float3 _112 = mapped;
        mapped.z = _112.yz.x;
        mapped.y = _112.yz.y;
    }
    if (flip.y) {
        float3 _122 = mapped;
        mapped.x = _122.yx.x;
        mapped.y = _122.yx.y;
    }
    if (flip.x) {
        float3 _132 = mapped;
        mapped.z = _132.yz.x;
        mapped.y = _132.yz.y;
    }
    return mapped;
}

}
