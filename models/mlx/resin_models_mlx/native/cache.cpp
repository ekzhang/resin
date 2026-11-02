// Engine-private mutable cache storage using MLX's Metal stream and hazards.
// This is an inference resource, not an immutable MLX tensor operation.
#include "common.h"
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <mlx/array.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {
using resin_models_mlx::native::checked_stream;
using resin_models_mlx::native::library_directory;

constexpr int kEpochElements = 8;

enum class Kind { Initialize, Read, Snapshot, Rows, Slice };

struct Storage {
  mx::Shape shape;
  mx::Dtype dtype;
  size_t size;
  std::optional<mx::array> arena;
  std::mutex mutex;
  size_t allocations = 0;

  Storage(const std::vector<int>& dimensions, mx::Dtype type)
      : shape(dimensions.begin(), dimensions.end()), dtype(type), size(1) {
    if (shape.empty() || (dtype != mx::bfloat16 && dtype != mx::float32)) {
      throw std::invalid_argument("CacheStorage requires nonempty shape and BF16/F32 dtype");
    }
    for (auto dimension : shape) {
      if (dimension < 1 || size > std::numeric_limits<size_t>::max() / size_t(dimension)) {
        throw std::invalid_argument("CacheStorage shape must be positive and fit size_t indexing");
      }
      size *= size_t(dimension);
    }
    if (size > std::numeric_limits<size_t>::max() / dtype.size()) {
      throw std::invalid_argument("CacheStorage byte count overflows size_t");
    }
  }

  mx::array& allocation(bool create = false) {
    std::lock_guard lock(mutex);
    if (!arena) {
      if (!create) {
        throw std::runtime_error("CacheStorage initial_epoch must precede cache access");
      }
      mx::array allocated(shape, dtype, nullptr, std::vector<mx::array>{});
      allocated.set_data(mx::allocator::malloc(size * dtype.size()));
      allocated.set_status(mx::array::Status::available);
      arena.emplace(std::move(allocated));
      ++allocations;
    }
    return *arena;
  }
};

void check_epoch(const mx::array& epoch) {
  if (epoch.dtype() != mx::uint32 || epoch.ndim() != 1 || epoch.size() != kEpochElements) {
    throw std::invalid_argument("CacheStorage epoch must be an eight-element uint32 array");
  }
}

class CacheOperation : public mx::Primitive {
 public:
  CacheOperation(mx::Stream stream, std::shared_ptr<Storage> storage, Kind kind,
                 uint64_t start = 0)
      : Primitive(stream), storage_(std::move(storage)), kind_(kind), start_(start) {}

  const char* name() const override { return "ResinPrivateCache"; }
  // Primitive defaults to is_equivalent=false. These resource operations must
  // not be CSE'd, differentiated, vectorized or graph-compiled.
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("CacheStorage has no CPU implementation");
  }

  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    auto& arena = storage_->allocation(kind_ == Kind::Initialize);
    auto& output = outputs[0];
    if (kind_ == Kind::Read) {
      // Private read view only. The eventual attention/gather primitive binds
      // this exact resource as its input. No memory access occurs here.
      output.copy_shared_buffer(arena);
      return;
    }
    output.set_data(mx::allocator::malloc(output.nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto library = device.get_library(MLX_METAL_LIBRARY_NAME, library_directory());
    auto suffix = storage_->dtype == mx::bfloat16 ? "_b2" : "_b4";
    if (kind_ == Kind::Initialize) {
      // Warm this owner's bounded byte-width catalog before serving. Neither
      // request rows nor context lengths select another pipeline.
      for (const std::string prefix : {"resin_cache_zero", "resin_cache_snapshot",
                                        "resin_cache_slice"}) {
        device.get_kernel(prefix + suffix, library);
      }
      device.get_kernel(std::string("resin_cache_rows") + suffix + "_i4", library);
      device.get_kernel(std::string("resin_cache_rows") + suffix + "_i8", library);
    }
    std::string kernel_name;
    switch (kind_) {
      case Kind::Initialize: kernel_name = "resin_cache_zero"; break;
      case Kind::Snapshot: kernel_name = "resin_cache_snapshot"; break;
      case Kind::Rows: kernel_name = "resin_cache_rows"; break;
      case Kind::Slice: kernel_name = "resin_cache_slice"; break;
      default: throw std::runtime_error("Invalid cache operation");
    }
    kernel_name += suffix;
    if (kind_ == Kind::Rows) {
      kernel_name += inputs[1].dtype() == mx::int32 ? "_i4" : "_i8";
    }
    auto kernel = device.get_kernel(kernel_name, library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    if (kind_ == Kind::Rows || kind_ == Kind::Slice) {
      const size_t base_inputs = kind_ == Kind::Rows ? 3 : 2;
      // Keep every completed reader in the writer's resource frontier. An
      // alias-only Depends node can lose this edge across encoder boundaries.
      // Slot 30 is outside the shaders' arguments; repeated binding records
      // every real input buffer without copying any of its contents.
      for (size_t i = base_inputs; i < inputs.size(); ++i) {
        encoder.set_input_array(inputs[i], 30);
      }
    }
    uint64_t work = storage_->size;
    if (kind_ == Kind::Initialize) {
      encoder.set_output_array(arena, 0);
      encoder.set_output_array(output, 1);
      encoder.set_bytes(storage_->size, 2);
    } else if (kind_ == Kind::Snapshot) {
      encoder.set_input_array(arena, 0);
      encoder.set_input_array(inputs[0], 1);
      encoder.set_output_array(output, 2);
      encoder.set_bytes(storage_->size, 3);
    } else {
      encoder.set_output_array(arena, 0);
      encoder.set_input_array(inputs[0], 1);
      encoder.set_output_array(output, 2);
      if (kind_ == Kind::Rows) {
        encoder.set_input_array(inputs[1], 3);
        encoder.set_input_array(inputs[2], 4);
        uint64_t width = storage_->shape.back();
        uint64_t count = inputs[1].size();
        uint64_t total_rows = storage_->size / width;
        encoder.set_bytes(width, 5);
        encoder.set_bytes(count, 6);
        encoder.set_bytes(total_rows, 7);
      } else {
        encoder.set_input_array(inputs[1], 3);
        uint64_t count = inputs[1].size();
        encoder.set_bytes(start_, 4);
        encoder.set_bytes(count, 5);
        work = count;
      }
    }
    // arena is a resource captured by the primitive, not an ordinary input or
    // output array. Keep its Data alive explicitly until the GPU completes.
    // add_temporary() is unsuitable: it removes resources from hazard tracking.
    encoder.get_command_buffer()->addCompletedHandler(
        [keep = arena.data_shared_ptr()](MTL::CommandBuffer*) {});
    auto threads = std::min<size_t>(256, kernel->maxTotalThreadsPerThreadgroup());
    // Width and rank-one index count are MLX int dimensions: both grid axes
    // fit uint32. Byte addresses remain 64-bit for large arenas.
    if (kind_ == Kind::Rows) {
      const uint64_t width = storage_->shape.back();
      const uint64_t count = inputs[1].size();
      const uint64_t columns = count ? std::max<uint64_t>(width, 8) : 8;
      encoder.dispatch_threads(MTL::Size(columns, std::max<uint64_t>(count, 1), 1),
                               MTL::Size(threads, 1, 1));
      return;
    }
    // Metal grid coordinates are uint32 per axis. Keep a bounded first
    // dimension and reconstruct the full linear address as ulong in MSL.
    work = std::max<uint64_t>(work, kEpochElements);
    constexpr uint64_t max_grid_width = 1 << 20;
    auto grid_width = std::min<uint64_t>(work, max_grid_width);
    auto grid_height = work / grid_width + (work % grid_width != 0);
    if (grid_height > UINT32_MAX) {
      throw std::invalid_argument("CacheStorage work exceeds the Metal grid capacity");
    }
    encoder.dispatch_threads(MTL::Size(grid_width, grid_height, 1),
                             MTL::Size(threads, 1, 1));
  }

 private:
  std::shared_ptr<Storage> storage_;
  Kind kind_;
  uint64_t start_;
};

class CacheStorage {
 public:
  CacheStorage(const std::vector<int>& shape, mx::Dtype dtype)
      : storage_(std::make_shared<Storage>(shape, dtype)) {}

  mx::array initial_epoch() {
    if (!initial_) {
      initial_ = operation(Kind::Initialize, {});
    }
    return *initial_;
  }
  mx::array read_epoch(const mx::array& epoch) {
    check_epoch(epoch);
    return operation(Kind::Read, {epoch});
  }
  mx::array snapshot(const mx::array& epoch) {
    check_epoch(epoch);
    auto result = operation(Kind::Snapshot, {epoch});
    pending_snapshots_.push_back(result);
    return result;
  }
  mx::array scatter_rows(const mx::array& epoch, const mx::array& indices,
                         const mx::array& updates, const std::vector<mx::array>& after) {
    check_epoch(epoch);
    if (indices.ndim() != 1 || (indices.dtype() != mx::int32 && indices.dtype() != mx::int64)) {
      throw std::invalid_argument("CacheStorage row indices must be rank-one int32/int64");
    }
    if (updates.dtype() != storage_->dtype || indices.size() > std::numeric_limits<size_t>::max() / size_t(storage_->shape.back())
        || updates.size() != indices.size() * size_t(storage_->shape.back())) {
      throw std::invalid_argument("CacheStorage row updates must match dtype and row payload size");
    }
    return operation(Kind::Rows, {epoch, mx::contiguous(indices), mx::contiguous(mx::reshape(updates, {-1}))}, 0, after);
  }
  mx::array write_slice(const mx::array& epoch, int64_t start,
                        const mx::array& updates, const std::vector<mx::array>& after) {
    check_epoch(epoch);
    if (start < 0 || uint64_t(start) > storage_->size || updates.size() > storage_->size - uint64_t(start)) {
      throw std::invalid_argument("CacheStorage slice is outside the allocation");
    }
    if (updates.dtype() != storage_->dtype) {
      throw std::invalid_argument("CacheStorage slice updates must match storage dtype");
    }
    return operation(Kind::Slice, {epoch, mx::contiguous(mx::reshape(updates, {-1}))}, start, after);
  }
  size_t nbytes() const { return storage_->size * storage_->dtype.size(); }
  uintptr_t identity() const { return reinterpret_cast<uintptr_t>(storage_.get()); }
  uintptr_t buffer_identity() const {
    std::lock_guard lock(storage_->mutex);
    return storage_->arena ? reinterpret_cast<uintptr_t>(storage_->arena->buffer().ptr()) : 0;
  }
  size_t allocation_count() const {
    std::lock_guard lock(storage_->mutex);
    return storage_->allocations;
  }
  std::vector<int> shape() const { return {storage_->shape.begin(), storage_->shape.end()}; }

 private:
  mx::array after_snapshots(const mx::array& epoch, std::vector<mx::array>& dependencies) {
    if (pending_snapshots_.empty()) return epoch;
    dependencies.insert(dependencies.end(), pending_snapshots_.begin(), pending_snapshots_.end());
    auto ordered = mx::depends({epoch}, pending_snapshots_)[0];
    return ordered;
  }
  mx::array operation(Kind kind, const std::vector<mx::array>& inputs, uint64_t start = 0,
                      const std::vector<mx::array>& after = {}) {
    auto stream = checked_stream();
    auto ordered_inputs = inputs;
    if (kind == Kind::Rows || kind == Kind::Slice) {
      auto dependencies = after;
      ordered_inputs[0] = after_snapshots(inputs[0], dependencies);
      ordered_inputs.insert(ordered_inputs.end(), dependencies.begin(), dependencies.end());
    }
    bool is_read = kind == Kind::Read || kind == Kind::Snapshot;
    auto result = mx::array(is_read ? storage_->shape : mx::Shape{kEpochElements},
                     is_read ? storage_->dtype : mx::uint32,
                     std::make_shared<CacheOperation>(stream, storage_, kind, start), ordered_inputs);
    if (kind == Kind::Rows || kind == Kind::Slice) pending_snapshots_.clear();
    return result;
  }
  std::shared_ptr<Storage> storage_;
  std::optional<mx::array> initial_;
  // Keep reader roots on the wrapper, avoiding a Storage→primitive→Storage cycle.
  std::vector<mx::array> pending_snapshots_;
};
}  // namespace

NB_MODULE(MLX_EXTENSION_NAME, module) {
  resin_models_mlx::native::bind_paged_attention(module);
  nb::class_<CacheStorage>(module, "CacheStorage")
      .def(nb::init<const std::vector<int>&, mx::Dtype>(), "shape"_a, "dtype"_a)
      .def("initial_epoch", &CacheStorage::initial_epoch)
      .def("read_epoch", &CacheStorage::read_epoch, "epoch"_a)
      .def("snapshot", &CacheStorage::snapshot, "epoch"_a)
      .def("scatter_rows", &CacheStorage::scatter_rows, "epoch"_a, "row_indices"_a, "updates"_a,
           nb::kw_only(), "after"_a = std::vector<mx::array>{})
      .def("write_slice", &CacheStorage::write_slice, "epoch"_a, "flat_start"_a, "updates"_a,
           nb::kw_only(), "after"_a = std::vector<mx::array>{})
      .def_prop_ro("nbytes", &CacheStorage::nbytes)
      .def_prop_ro("identity", &CacheStorage::identity)
      .def_prop_ro("buffer_identity", &CacheStorage::buffer_identity)
      .def_prop_ro("shape", &CacheStorage::shape)
      .def_prop_ro("allocation_count", &CacheStorage::allocation_count);
}
