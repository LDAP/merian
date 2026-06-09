#extension GL_EXT_scalar_block_layout           : enable

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout (constant_id = 2) const float COMPONENT = 0;

layout(set = 0, binding = 0) uniform sampler2D img_src;

layout(set = 0, binding = 1, scalar) buffer restrict buf_result {
    float result[];
};

layout(set = 0, binding = 2, scalar) buffer restrict buf_hist {
    uint histogram[];
};

layout(push_constant) uniform PushStruct {
    float min;
    float max;
} params;
