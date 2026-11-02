#pragma once
#include <nanobind/nanobind.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>

namespace resin_models_mlx::native {
inline mlx::core::Stream checked_stream() {
  namespace mx = mlx::core;
  // Probe tracing without evaluating data or retaining a stream across threads.
  static thread_local mx::array probe = [] {
    mx::array value(mx::Shape{}, mx::bool_, nullptr, {});
    value.set_tracer(true);
    return value;
  }();
  if (probe.is_tracer()) {
    throw std::invalid_argument("Native cache resources do not support compile/grad/vmap/export");
  }
  auto stream = mx::to_stream({});
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument("Native cache resources require the current GPU stream");
  }
  return stream;
}
inline std::string library_directory() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&library_directory), &info)) {
    throw std::runtime_error("Cannot locate native cache Metal library");
  }
  return std::filesystem::path(info.dli_fname).parent_path().string();
}
void bind_paged_attention(nanobind::module_& module);
}  // namespace resin_models_mlx::native
