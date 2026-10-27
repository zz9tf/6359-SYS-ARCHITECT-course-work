"""NVML-based GPU monitor: energy, power, device memory and SM clock during a measured region."""
import threading
import time

import pynvml

_INIT = False


def _handle(index: int = 0):
    global _INIT
    if not _INIT:
        pynvml.nvmlInit()
        _INIT = True
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def device_memory_used(index: int = 0) -> int:
    """Device memory in use as seen by the driver (includes CUDA context and other processes)."""
    return pynvml.nvmlDeviceGetMemoryInfo(_handle(index)).used


def gpu_info(index: int = 0) -> dict:
    h = _handle(index)
    return {
        "gpu_name": pynvml.nvmlDeviceGetName(h),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "memory_total_bytes": pynvml.nvmlDeviceGetMemoryInfo(h).total,
        "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
        "max_sm_clock_mhz": pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM),
        "max_mem_clock_mhz": pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM),
    }


class GpuMonitor:
    """Context manager that integrates GPU energy and samples memory/power/clock in a thread.

    Energy comes from the driver's cumulative energy counter (mJ); the sampled power trace is kept
    as a cross-check. Sampling every 50 ms keeps the thread's GIL footprint negligible.
    """

    def __init__(self, index: int = 0, interval: float = 0.05):
        self.h = _handle(index)
        self.interval = interval

    def __enter__(self):
        self.samples = []
        self._stop = threading.Event()
        self._sample()
        self.e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _sample(self):
        self.samples.append((
            pynvml.nvmlDeviceGetMemoryInfo(self.h).used,
            pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
            pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
        ))

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()
        self.t1 = time.perf_counter()
        self.e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
        self._sample()
        self.temp_c = pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)
        return False

    @property
    def result(self) -> dict:
        dur = self.t1 - self.t0
        energy = (self.e1 - self.e0) / 1000.0
        mems, powers, clocks = zip(*self.samples)
        return {
            "energy_j": energy,
            "avg_power_w": energy / dur if dur > 0 else float("nan"),
            "sampled_avg_power_w": sum(powers) / len(powers),
            "nvml_peak_used_bytes": max(mems),
            "avg_sm_clock_mhz": sum(clocks) / len(clocks),
            "temp_end_c": self.temp_c,
        }
