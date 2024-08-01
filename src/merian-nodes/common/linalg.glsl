#ifndef _LINALG_H_
#define _LINALG_H_


// Returns the Moore-Penrose inverse also known as the pseudoinverse A+ = (A^TA)^-1 * A^T.
// A+ * b is the solution that minimizes || Ax - b ||.
// 
// For A = (v(2) - v(0), v(1) - v(0)) this is also the worldspace derivative of the barycentric coordinates.
// n = cross(A[0], A[1])
mat3x2 pseudoinverse(const mat2x3 A, in vec3 n) {
    // const mat3x2 A_T = transpose(A);
    // return inverse(A_T * A) * A_T;
    
    // this is a more efficient solution of the above:
    n /= dot(n, n);
    return transpose(mat2x3(cross(A[1], n), cross(n, A[0])));
}

mat3x2 pseudoinverse(const mat2x3 A) {
    return pseudoinverse(A, cross(A[0], A[1]));
}

void pseudoinverse(const vec3 du, const vec3 dv, const vec3 cross_dudv, out vec3 du_dx, out vec3 dv_dx) {
    const vec3 nt = cross_dudv / dot(cross_dudv, cross_dudv);
    du_dx = cross(dv, nt);
    dv_dx = cross(nt, du);
}

// du = v(2) - v(0), dv = v(1) - v(0)
void pseudoinverse(const vec3 du, const vec3 dv, out vec3 du_dx, out vec3 dv_dx) {
    pseudoinverse(du, dv, cross(du, dv), du_dx, dv_dx);
}

#endif // _LINALG_H_
