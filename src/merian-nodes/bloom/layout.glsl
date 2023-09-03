layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(constant_id = 2) const int MODE = 0;

layout(set = 0, binding = 0) uniform sampler2D img_src;
layout(set = 0, binding = 1) uniform writeonly image2D img_out;
layout(set = 0, binding = 2, rgba16f) uniform image2D img_interm;

layout(push_constant) uniform PushStruct {
    float threshold;
    float strength;
} pc;
