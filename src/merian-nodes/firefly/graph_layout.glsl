layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D img_irr;
layout(set = 0, binding = 1) uniform sampler2D img_moments;

layout(set = 0, binding = 2) uniform writeonly image2D img_irr_out;
layout(set = 0, binding = 3) uniform writeonly image2D img_moments_out;
