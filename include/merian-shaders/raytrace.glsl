#version 460

#extension GL_GOOGLE_include_directive : require


#ifndef _MERIAN_SHADERS_RAYTRACE_H_
#define _MERIAN_SHADERS_RAYTRACE_H_

#define _rq_get_t(ray_query, commited) rayQueryGetIntersectionTEXT(ray_query, commited)
#define _rq_instance_id(ray_query, commited) rayQueryGetIntersectionInstanceIdEXT(ray_query, commited)
#define _rq_primitive_index(ray_query, commited) rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, commited)
#define _rq_barycentrics(ray_query, commited) vec3(1.0 - dot(rayQueryGetIntersectionBarycentricsEXT(ray_query, commited), vec2(1)), rayQueryGetIntersectionBarycentricsEXT(ray_query, commited))

#define rq_get_t(ray_query) _rq_get_t(ray_query, true)
#define rq_instance_id(ray_query) _rq_instance_id(ray_query, true)
#define rq_primitive_index(ray_query) _rq_primitive_index(ray_query, true)  
#define rq_barycentrics(ray_query) _rq_barycentrics(ray_query, true)

#define rq_get_t_uc(ray_query) _rq_get_t(ray_query, false)
#define rq_instance_id_uc(ray_query) _rq_instance_id(ray_query, false)
#define rq_primitive_index_uc(ray_query) _rq_primitive_index(ray_query, false)  
#define rq_barycentrics_uc(ray_query) _rq_barycentrics(ray_query, false)

struct IntersectionInfo {
    uint instance_id;
    // primitive index in the instance
    uint primitive_index;
    vec3 barycentrics;
    // distance to intersection
    float t;
};

// The the intersection info for a commited intersection
void get_intersection_info_commited(const rayQueryEXT ray_query, out IntersectionInfo info) {
    info.instance_id = rayQueryGetIntersectionInstanceIdEXT(ray_query, true);
    info.primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
    info.barycentrics.yz = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);
    info.barycentrics.x = 1.0 - info.barycentrics.z - info.barycentrics.y;
    info.t = rayQueryGetIntersectionTEXT(ray_query, true);
}

// The the intersection info for an uncommited intersection
void get_intersection_info_uncommited(const rayQueryEXT ray_query, out IntersectionInfo info) {
    info.instance_id = rayQueryGetIntersectionInstanceIdEXT(ray_query, false);
    info.primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, false);
    info.barycentrics.yz = rayQueryGetIntersectionBarycentricsEXT(ray_query, false);
    info.barycentrics.x = 1.0 - info.barycentrics.z - info.barycentrics.y;
    info.t = rayQueryGetIntersectionTEXT(ray_query, false);
}

#endif
