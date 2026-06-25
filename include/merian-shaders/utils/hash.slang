namespace merian {

// Jarzynski et al. From Hash Functions for GPU Rendering:

// Many of the hashes that fall along the Pareto frontier were designed to be executed on the GPU
// with neighboring seeds. Chained PRNGs with a state, such as city, murmur3, and ranlim32, may be
// great PRNGs [Pike and Alakuijala 2011; Appleby 2008; Press et al. 2007] when used sequentially,
// but when used in parallel with neighboring seeds, they fall short. We conclude by providing a
// spanning set of exemplar algorithms for each usage category, ordered from fastest to highest
// quality.

// Fastest to highest quality
// (1 → 1): iqint1, pcg, xxhash32, pcg3d, pcg4d
// (2 → 1): iqint3, xxhash32_multibyte2, pcg3d, pcg4d
// (3 → 1): pcg3d16, xxhash32_multibyte3, pcg3d, pcg4d
// (4 → 1): xxhash32_multibyte4, pcg4d
// (N → N ): pcg3d, pcg4d

// (1 → 1):

// Integer Hash - I
// - Inigo Quilez, Integer Hash - I, 2017
//   https://www.shadertoy.com/view/llGSzw
uint iqint1(uint n)
{
    // integer hash copied from Hugo Elias
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;

    return n;
}

// https://www.pcg-random.org/
uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/**
 * https://github.com/Cyan4973/xxHash
 */
uint xxhash32(uint p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p + PRIME32_5;
    h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2*(h32^(h32 >> 15));
    h32 = PRIME32_3*(h32^(h32 >> 13));
    return h32^(h32 >> 16);
}

// (2 → 1):

// Integer Hash - III
// - Inigo Quilez, Integer Hash - III, 2017
//   https://www.shadertoy.com/view/4tXyWN
uint iqint3(uint2 x)
{
    uint2 q = 1103515245U * ( (x>>1U) ^ (x.yx   ) );
    uint  n = 1103515245U * ( (q.x  ) ^ (q.y>>3U) );

    return n;
}

uint xxhash32(uint2 p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
	const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p.y + PRIME32_5 + p.x*PRIME32_3;
    h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2*(h32^(h32 >> 15));
    h32 = PRIME32_3*(h32^(h32 >> 13));
    return h32^(h32 >> 16);
}


// (3 → 1):

// https://www.pcg-random.org/
uint pcg3d16(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.x += v.y * v.z;
    return v.x;
}

uint xxhash32(uint3 p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
	const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
	uint h32 =  p.z + PRIME32_5 + p.x*PRIME32_3;
	h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
	h32 += p.y * PRIME32_3;
	h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2*(h32^(h32 >> 15));
    h32 = PRIME32_3*(h32^(h32 >> 13));
    return h32^(h32 >> 16);
}

// (3 → 3):

// https://www.pcg-random.org/
uint3 pcg3d(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// (4 → 1):

uint xxhash32(uint4 p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
	const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
	uint h32 =  p.w + PRIME32_5 + p.x*PRIME32_3;
	h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
	h32 += p.y * PRIME32_3;
	h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
	h32 += p.z * PRIME32_3;
	h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2*(h32^(h32 >> 15));
    h32 = PRIME32_3*(h32^(h32 >> 13));
    return h32^(h32 >> 16);
}

// https://www.pcg-random.org/
uint pcg4d16(uint4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v.x += v.y * v.w;
    return v.x;
}

// (4 → 4):

// https://www.pcg-random.org/
uint4 pcg4d(uint4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

// -----------------------------------
// VERY FAST BUT BAD!

// linear congruential generator
uint lcg(uint p)
{
    return p * 1664525u + 1013904223u;
}

// 128-bit xorshift
//  - Marsaglia, Xorshift RNGs, Journal of Statistical Software, v8n14, 2003
uint xorshift128(uint4 v)
{
    v.w ^= v.w << 11u;
    v.w ^= v.w >> 8u;
    v = v.wxyz;
    v.x ^= v.y;
    v.x ^= v.y >> 19u;
    return v.x;
}



// 32-bit xorshift
//  - Marsaglia, Xorshift RNGs, Journal of Statistical Software, v8n14, 2003
uint xorshift32(uint v)
{
    v ^= v << 13u;
    v ^= v >> 17u;
    v ^= v << 5u;
    return v;
}

// Teschner, Heidelberger, Müller, Pomeranets & Gross, "Optimized Spatial Hashing for Collision Detection of Deformable Objects" (VMV 2003)
// Used by Müller et al. (2022): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
// Cheap but leads to lots of collisions.
uint teschner(uint3 v) {
	const uint PRIME_X = 73856093;
	const uint PRIME_Y = 19349663;
	const uint PRIME_Z = 83492791;

	return ((v.x * PRIME_X) ^ (v.y * PRIME_Y) ^ (v.z * PRIME_Z));
}

// -----------------------------------
// Other

// Used by NVIDIA SHARCs hash grid
// https://github.com/NVIDIA-RTX/SHARC/blob/e19ccacd511f42a3df6f850052d508c13c9e9737/include/HashGridCommon.h#L105
// https://burtleburtle.net/bob/hash/integer.html
uint wang(uint a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

}



 




 



 




 



 




 


 




 



 




 


 




 



 




 



 




 



 




 



 




 



 




 
