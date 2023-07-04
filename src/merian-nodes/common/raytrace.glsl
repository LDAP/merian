#ifndef _RAYTRACE_H_
#define _RAYTRACE_H_

struct IntersectionInfo {
    uint instance_custom_index;
    // primitive index in the instance
    uint primitive_index;
    vec3 barycentrics;
    // distance to intersection
    float t;
};

// The the intersection info for a commited intersection
void get_intersection_info_commited(const rayQueryEXT ray_query, inout IntersectionInfo info) {
    info.instance_custom_index = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true);
    info.primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
    info.barycentrics.yz = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);
    info.barycentrics.x = 1.0 - info.barycentrics.z - info.barycentrics.y;
    info.t = rayQueryGetIntersectionTEXT(ray_query, true);
}

// The the intersection info for an uncommited intersection
void get_intersection_info_uncommited(const rayQueryEXT ray_query, inout IntersectionInfo info) {
    info.instance_custom_index = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, false);
    info.primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, false);
    info.barycentrics.yz = rayQueryGetIntersectionBarycentricsEXT(ray_query, false);
    info.barycentrics.x = 1.0 - info.barycentrics.z - info.barycentrics.y;
    info.t = rayQueryGetIntersectionTEXT(ray_query, false);
}

#endif
