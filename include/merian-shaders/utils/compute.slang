#pragma once

namespace merian {

// Improves L2 cache locality by swizzling workgroup IDs so that they wrap after N workgroups in x
// direction.
//
// Reference:
// https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/
// Reference:
// https://github.com/LouisBavoil/ThreadGroupIDSwizzling/blob/master/ThreadGroupTilingX.hlsl
uint2 Bavoil_DispatchThreadID(const uint2 groupID, const uint2 groupThreadID, const uint N = 16) {
    const uint2 numGroups = WorkgroupCount().xy;

    const uint tileSize = N * numGroups.y;
    const uint flatID = numGroups.x * groupID.y + groupID.x;
    const uint tileID = flatID / tileSize;
    const uint localCTA = flatID % tileSize;

    // last tile may be narrower if numGroups.x is not a multiple of N
    const uint totalPerfect = (numGroups.x / N) * tileSize;
    const uint tileW = (flatID >= totalPerfect) ? (numGroups.x % N) : N;

    const uint swizzledFlat = tileID * N + (localCTA / tileW) * numGroups.x + (localCTA % tileW);
    const uint2 swizzledGroup = uint2(swizzledFlat % numGroups.x, swizzledFlat / numGroups.x);

    return WorkgroupSize().xy * swizzledGroup + groupThreadID;
}

}
