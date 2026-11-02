// Bit-preserving writes to Engine-owned storage; no floating-point arithmetic.
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void cache_zero(device T* arena [[buffer(0)]],
                       device uint* epoch [[buffer(1)]],
                       constant ulong& count [[buffer(2)]], uint3 position [[thread_position_in_grid]],
                       uint3 grid [[threads_per_grid]]) {
  const ulong tid = ulong(position.x) + ulong(position.y) * ulong(grid.x);
  if (tid < count) arena[tid] = T(0);
  if (tid < 8) epoch[tid] = 0;
}
template <typename T>
kernel void cache_snapshot(device const T* arena [[buffer(0)]],
                           device const uint* epoch [[buffer(1)]],
                           device T* snapshot [[buffer(2)]],
                           constant ulong& count [[buffer(3)]], uint3 position [[thread_position_in_grid]],
                       uint3 grid [[threads_per_grid]]) {
  const ulong tid = ulong(position.x) + ulong(position.y) * ulong(grid.x);
  if (tid < count) snapshot[tid] = arena[tid];
}
template <typename T, typename Index>
kernel void cache_rows(device T* arena [[buffer(0)]],
                         device const uint* old_epoch [[buffer(1)]],
                         device uint* epoch [[buffer(2)]],
                         device const Index* indices [[buffer(3)]],
                         device const T* updates [[buffer(4)]],
                         constant ulong& width [[buffer(5)]],
                         constant ulong& count [[buffer(6)]],
                         constant ulong& total_rows [[buffer(7)]],
                         uint3 position [[thread_position_in_grid]],
                         uint3 grid [[threads_per_grid]]) {
  const ulong column = ulong(position.x);
  const ulong row = ulong(position.y);
  if (column < width && row < count) {
    const long target = long(indices[row]);
    if (target >= 0 && ulong(target) < total_rows)
      arena[ulong(target) * width + column] = updates[row * width + column];
  }
  if (row == 0 && column < 8) epoch[column] = old_epoch[column] + 1;
}
template <typename T>
kernel void cache_slice(device T* arena [[buffer(0)]],
                        device const uint* old_epoch [[buffer(1)]],
                        device uint* epoch [[buffer(2)]],
                        device const T* updates [[buffer(3)]],
                        constant ulong& start [[buffer(4)]],
                        constant ulong& count [[buffer(5)]], uint3 position [[thread_position_in_grid]],
                       uint3 grid [[threads_per_grid]]) {
  const ulong tid = ulong(position.x) + ulong(position.y) * ulong(grid.x);
  if (tid < count) arena[start + tid] = updates[tid];
  if (tid < 8) epoch[tid] = old_epoch[tid] + 1;
}
#define INSTANTIATE(T, SUFFIX) \
  template [[host_name("resin_cache_zero_" #SUFFIX)]] kernel void cache_zero<T>(device T*, device uint*, constant ulong&, uint3, uint3); \
  template [[host_name("resin_cache_snapshot_" #SUFFIX)]] kernel void cache_snapshot<T>(device const T*, device const uint*, device T*, constant ulong&, uint3, uint3); \
  template [[host_name("resin_cache_slice_" #SUFFIX)]] kernel void cache_slice<T>(device T*, device const uint*, device uint*, device const T*, constant ulong&, constant ulong&, uint3, uint3); \
  template [[host_name("resin_cache_rows_" #SUFFIX "_i4")]] kernel void cache_rows<T,int>(device T*, device const uint*, device uint*, device const int*, device const T*, constant ulong&, constant ulong&, constant ulong&, uint3, uint3); \
  template [[host_name("resin_cache_rows_" #SUFFIX "_i8")]] kernel void cache_rows<T,long>(device T*, device const uint*, device uint*, device const long*, device const T*, constant ulong&, constant ulong&, constant ulong&, uint3, uint3);
INSTANTIATE(ushort, b2)
INSTANTIATE(uint, b4)
