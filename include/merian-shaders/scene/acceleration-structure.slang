#pragma once

namespace merian {

// ---------------------------------------
// Acceleration / Ray Query Interface

interface IRayQuery {
    uint CommittedStatus();
    uint CommittedInstanceID(); // user-defined ID assigned to the instance when the TLAS was built,
                                // CommitedInstanceIndex() is the fixed instance index.
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

// ---------------------------------------

struct NullRayQuery : IRayQuery {
    uint CommittedStatus() {
        return COMMITTED_NOTHING;
    }
    uint CommittedInstanceID() {
        return 0;
    }
    uint CommittedPrimitiveIndex() {
        return 0;
    }
    uint CommittedGeometryIndex() {
        return 0;
    }
    float2 CommittedRayBarycentrics() {
        return float2(0);
    }

    uint CandidateType() {
        return 0;
    }
    uint CandidateInstanceID() {
        return 0;
    }
    uint CandidatePrimitiveIndex() {
        return 0;
    }
    uint CandidateGeometryIndex() {
        return 0;
    }
    float2 CandidateRayBarycentrics() {
        return float2(0);
    }

    [mutating]
    bool Proceed() {
        return false;
    }
    [mutating]
    void CommitNonOpaqueTriangleHit() {}
    [mutating]
    void CommitProceduralPrimitiveHit(float t) {}
}

struct NullAccelerationStructure : AccelerationStructure {
    typealias RayQueryImpl = NullRayQuery;

    RayQueryImpl trace_ray(const RayDesc ray,
                           const uint ray_flags = RAY_FLAG_NONE,
                           const uint instances_mask = 0xff) {
        NullRayQuery rq;
        return rq;
    }
}

}

    
    
    
        
    
    
        
    
        
    
        
    
        
    
        
    
        
    
        
    
        
    
        
    
        
