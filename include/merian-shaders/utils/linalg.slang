float3x2 pseudoinverse(float2x3 A, float3 n)
{
    n /= dot(n, n).xxx;
    return transpose(float2x3(float3(cross(A[1], n)), float3(cross(n, A[0]))));
}

float3x2 pseudoinverse(float2x3 A)
{
    float3 param = cross(A[0], A[1]);
    float3x2 _68 = pseudoinverse(A, param);
    return _68;
}

void pseudoinverse(float3 du, float3 dv, float3 cross_dudv, out float3 du_dx, out float3 dv_dx)
{
    float3 nt = cross_dudv / dot(cross_dudv, cross_dudv).xxx;
    du_dx = cross(dv, nt);
    dv_dx = cross(nt, du);
}

void pseudoinverse(float3 du, float3 dv, out float3 du_dx, out float3 dv_dx)
{
    float3 param;
    float3 param_1;
    pseudoinverse(du, dv, cross(du, dv), param, param_1);
    du_dx = param;
    dv_dx = param_1;
}

