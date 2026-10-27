"""Result files and environment capture."""
import datetime as dt
import json
import platform
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def append_jsonl(path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), **row}
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def env_info() -> dict:
    import torch

    from .monitor import gpu_info

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
    }
    info.update({pkg: ver(pkg) for pkg in ("transformers", "bitsandbytes", "vllm", "accelerate")})
    info.update(gpu_info())
    return info
