"""A DRAM LRU for immutable objects, budgeting both key and payload bytes."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain


class MemoryCacheStorage:
    def __init__(self, max_bytes=1 << 30):
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes, self.nbytes = max_bytes, 0
        self.objects, self.executor = OrderedDict(), ThreadPoolExecutor(1)

    def lookup(self, keys):
        return self.executor.submit(lambda: tuple(key in self.objects for key in keys))

    def load(self, keys, buffers):
        def read():
            for key, regions in zip(keys, buffers, strict=True):
                data = self.objects[key]
                if len(data) != sum(r.nbytes for r in regions):
                    return False
                self.objects.move_to_end(key)
                offset = 0
                for region in regions:
                    chunk = data[offset : offset + region.nbytes]
                    if not chunk:
                        continue
                    if hasattr(region, "copy_"):
                        import torch

                        with torch.inference_mode():
                            region.reshape(-1).copy_(
                                torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                            )
                    else:
                        memoryview(region).cast("B")[:] = chunk
                    offset += region.nbytes
            return True

        return self._submit(read, buffers)

    def store(self, keys, buffers):
        def write():
            success = True
            for key, regions in zip(keys, buffers, strict=True):
                if key in self.objects:
                    continue
                size = len(key) + sum(r.nbytes for r in regions)
                if size > self.max_bytes:
                    success = False
                    continue
                data = b"".join(
                    r.detach().cpu().numpy().tobytes()
                    if hasattr(r, "detach")
                    else bytes(r)
                    for r in regions
                )
                while self.nbytes + size > self.max_bytes:
                    old_key, old_data = self.objects.popitem(last=False)
                    self.nbytes -= len(old_key) + len(old_data)
                self.objects[key] = data
                self.nbytes += size
            return success

        return self._submit(write, buffers)

    def _submit(self, action, buffers):
        regions = chain.from_iterable(buffers)
        devices = {r.device for r in regions if getattr(r, "is_cuda", False)}
        ready = {}
        if devices:
            import torch

            ready = {d: torch.cuda.current_stream(d).record_event() for d in devices}

        def run():
            streams = []
            try:
                for device, event in ready.items():
                    stream = torch.cuda.Stream(device=device)
                    streams.append(stream)
                    stream.wait_event(event)
                    torch.cuda.set_stream(stream)
                result = action()
            except Exception:
                result = False
            try:
                for stream in streams:
                    stream.synchronize()
            except Exception:
                return None
            return result

        return self.executor.submit(run)

    def close(self):
        self.executor.shutdown(wait=True)
