#pragma once

namespace merian {

// ---------------------------------------
// Acceleration / Ray Query Interface

interface IRayQuery {
    uint CommittedStatus();
    uint CommittedInstanceID();
    uint CommittedPrimitiveIndex();
    uint CommittedGeometryIndex();
    float2 CommittedRayBarycentrics();

    uint CandidateType();
    uint CandidateInstanceID();
    uint CandidatePrimitiveIndex();
    uint CandidateGeometryIndex();
    float2 CandidateRayBarycentrics();

    [mutating]
    bool Proceed();
    [mutating]
    void CommitNonOpaqueTriangleHit();
    [mutating]
    void CommitProceduralPrimitiveHit(float t);

    // ----------------------------

    float3 CommittedRayBarycentrics2() {
        float3 barycentrics;
        barycentrics.xy = CommittedRayBarycentrics();
        barycentrics.z = 1.0 - barycentrics.x - barycentrics.y;
        return barycentrics;
    }

    float3 CandidateRayBarycentrics2() {
        float3 barycentrics;
        barycentrics.xy = CandidateRayBarycentrics();
        barycentrics.z = 1.0 - barycentrics.x - barycentrics.y;
        return barycentrics;
    }
}

interface AccelerationStructure {
    associatedtype RayQueryImpl : IRayQuery;

    RayQueryImpl trace_ray(const RayDesc ray,
                           const uint ray_flags = RAY_FLAG_NONE,
                           const uint instances_mask = 0xff);
}

// ---------------------------------------

// extend the built in RayQuery to use it without wrapper
extension RayQuery : IRayQuery {}

struct HWAccelerationStructure : AccelerationStructure {
    RaytracingAccelerationStructure as;

    // -----------------------------------------

    typealias RayQueryImpl = RayQuery;

    RayQueryImpl trace_ray(const RayDesc ray,
                           const uint ray_flags = RAY_FLAG_NONE,
                           const uint instances_mask = 0xff) {
        RayQuery rq;
        rq.TraceRayInline(as, ray_flags, instances_mask, ray);
        return rq;
    }
}

}
