#ifndef _MERIAN_SHADERS_GBUFFER_H_
#define _MERIAN_SHADERS_GBUFFER_H_

#ifdef __cplusplus
// c++ only
namespace merian_nodes {
#else
// GLSL only
// clang-format off

#extension GL_EXT_shader_explicit_arithmetic_types : enable
// #extension GL_EXT_scalar_block_layout : enable

#define MAKE_GBUFFER_READONLY_LAYOUT(set, binding, gbuffer_binding_name)     layout(set, binding)           uniform                      usampler2D gbuffer_binding_name
#define MAKE_GBUFFER_WRITEONLY_LAYOUT(set, binding, gbuffer_binding_name)    layout(set, binding, rgba32ui) uniform writeonly   restrict uimage2D   gbuffer_binding_name
#define MAKE_GBUFFER_READWRITE_LAYOUT(set, binding, gbuffer_binding_name)    layout(set, binding, rgba32ui) uniform             restrict uimage2D   gbuffer_binding_name

// u32vec4(
//    .x = encoded normal of pixel
//    .y = linear distance from camera to pixel,
//    .z = dlinear_z / dipos in depth / pixel,
//    .w = camera velocity in ray direction
// )
#define GBuffer u32vec4

#define gbuffer_new() u32vec4(0)

struct DecodedGBuffer {
    vec3 normal;
    float linear_z;
    f16vec2 grad_z;
    float vel_z;
};

#define _gbuffer_get_enc_normal(gbuffer_variable_name)      gbuffer_variable_name.x
#define _gbuffer_get_enc_linear_z(gbuffer_variable_name)    gbuffer_variable_name.y
#define _gbuffer_get_enc_grad_z(gbuffer_variable_name)      gbuffer_variable_name.z
#define _gbuffer_get_enc_vel_z(gbuffer_variable_name)       gbuffer_variable_name.w

#define _gbuffer_decode_normal(encoded_gbuffer)     geo_decode_normal(encoded_gbuffer.x)
#define _gbuffer_decode_linear_z(encoded_gbuffer)   uintBitsToFloat(encoded_gbuffer.y)
#define _gbuffer_decode_grad_z(encoded_gbuffer)     unpackFloat2x16(encoded_gbuffer.z)
#define _gbuffer_decode_vel_z(encoded_gbuffer)      uintBitsToFloat(encoded_gbuffer.w)

#define _gbuffer_encode_normal(normal)     geo_encode_normal(normal)
#define _gbuffer_encode_linear_z(linear_z) floatBitsToUint(linear_z)
#define _gbuffer_encode_grad_z(grad_z)     packFloat2x16(grad_z)
#define _gbuffer_encode_vel_z(vel_z)       floatBitsToUint(vel_z)


#define gbuffer_load_pixel(gbuffer_binding_name, pixel) texelFetch(gbuffer_binding_name, pixel, 0)
#define gbuffer_store_pixel(gbuffer_binding_name, pixel, encoded_gbuffer) imageStore(gbuffer_binding_name, pixel, encoded_gbuffer)


#define gbuffer_get_normal(gbuffer) _gbuffer_decode_normal(gbuffer)  
#define gbuffer_get_linear_z(gbuffer) _gbuffer_decode_linear_z(gbuffer)
#define gbuffer_get_grad_z(gbuffer) _gbuffer_decode_grad_z(gbuffer)  
#define gbuffer_get_vel_z(gbuffer) _gbuffer_decode_vel_z(gbuffer)

#define gbuffer_get_decoded(encoded_gbuffer) DecodedGBuffer(gbuffer_get_normal(encoded_gbuffer), gbuffer_get_linear_z(encoded_gbuffer), gbuffer_get_grad_z(encoded_gbuffer), gbuffer_get_vel_z(encoded_gbuffer))

#define gbuffer_encode(normal, linear_z, grad_z, vel_z) GBuffer(_gbuffer_encode_normal(normal), _gbuffer_encode_linear_z(linear_z), _gbuffer_encode_grad_z(grad_z), _gbuffer_encode_vel_z(vel_z))
#define gbuffer_encode_decoded(decoded_gbuffer) GBuffer(_gbuffer_encode_normal(decoded_gbuffer.normal), _gbuffer_encode_linear_z(decoded_gbuffer.linear_z), _gbuffer_encode_grad_z(decoded_gbuffer.grad_z), _gbuffer_encode_vel_z(decoded_gbuffer.vel_z))


#define gbuffer_get_normal_pixel(gbuffer_binding_name, pixel) gbuffer_get_normal(gbuffer_load_pixel(gbuffer_binding_name, pixel))
#define gbuffer_get_linear_z_pixel(gbuffer_binding_name, pixel) gbuffer_get_linear_z(gbuffer_load_pixel(gbuffer_binding_name, pixel))
#define gbuffer_get_grad_z_pixel(gbuffer_binding_name, pixel) gbuffer_get_grad_z(gbuffer_load_pixel(gbuffer_binding_name, pixel))
#define gbuffer_get_vel_z_pixel(gbuffer_binding_name, pixel) gbuffer_get_vel_z(gbuffer_load_pixel(gbuffer_binding_name, pixel))

#define gbuffer_get_decoded_pixel(gbuffer_binding_name, pixel) gbuffer_get_decoded(gbuffer_load_pixel(gbuffer_binding_name, pixel))

// clang-format on

#endif

// struct GBuffer {
//     // encoded normal of pixel
//     uint32_t enc_normal;
//     // linear distance from camera to pixel
//     float linear_z;
//     // dlinear_z / dipos in depth / pixel
//     f16vec2 grad_z;
//     // camera velocity in ray direction
//     float vel_z;
// };

#ifdef __cplusplus
}
#endif

#endif
