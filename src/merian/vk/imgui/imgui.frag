#version 450

// ImGui colors are authored display-encoded; an sRGB attachment encodes on write, so the
// pipeline sets this to linearize them first (otherwise they wash out).
layout(constant_id = 0) const uint CONVERT_SRGB = 0;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) out vec4 out_color;

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

void main() {
    vec4 color = in_color;
    if (CONVERT_SRGB != 0) {
        color.rgb = srgb_to_linear(color.rgb);
    }
    out_color = color * texture(tex, in_uv);
}
