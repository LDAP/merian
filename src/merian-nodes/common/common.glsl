#ifndef _COMMON_H_
#define _COMMON_H_

#define M_PI   3.14159265358979323846
#define TWO_PI 6.283185307179586
#define INV_PI 0.3183098861837907
#define INV_SQRT_TWO_PI 0.3989422804014327

// returns 1/x if x > 0 else 1.
#define SAFE_RECIPROCAL(x) (x > 0. ? 1. / x : 1.0)

#endif // _COMMON_H_
