#ifndef _INTERSECT_H_
#define _INTERSECT_H_

// Returns t such that ray_origin + t * ray_dir = intersection
#define intersect_ray_plane(plane_origin, plane_normal, ray_origin, ray_dir) (dot(plane_normal, plane_origin - ray_origin) / dot(plane_normal, ray_dir))


#endif
