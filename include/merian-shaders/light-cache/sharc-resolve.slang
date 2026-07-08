import merian_shaders.light_cache.sharc;

using merian;

// Keep in sync with Sharc::RESOLVE_LOCAL_SIZE_X on the host side.
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id: SV_DispatchThreadID, ParameterBlock<Sharc> sharc) {
    sharc.resolve(thread_id.x);
}
