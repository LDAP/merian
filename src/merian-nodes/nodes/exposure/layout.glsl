#extension GL_EXT_scalar_block_layout       : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D img_src;
layout(set = 0, binding = 1) uniform writeonly image2D img_out;

layout(set = 0, binding = 2, scalar) buffer restrict buf_hist {
    uint histogram[];
};
layout(set = 0, binding = 3, scalar) buffer restrict buf_lum {
    float luminance[];
};

layout(push_constant) uniform PushStruct {
    int automatic;

    float iso;
    float q;

    float shutter_time;
    float aperature;

    float K;
    float speed_up;
    float speed_down;
    float timediff;
    int reset;
    float min_log_histogram;
    float max_log_histogram;
    int metering;
    float min_exposure;
    float max_exposure;

} params;
