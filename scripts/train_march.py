"""Run the recovered March 11 benchmark under CPython 3.12."""
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "src" / "historical"
MODULES = (
    ("TransRheo_highfit_multitask", "march_backbone", "4a3be8289511ebc69542d7aeb5731384a278b76e26c4f9747b5baa55801030f3"),
    ("llm_transrheo_prior_correction_pipeline", "march_correction", "7ed72f14d357bd9b6326c0e3b8fb2e079db17b31b8e23e5db7e65a3541308705"),
    ("einn_march_benchmark", "march_benchmark", "a93ad1f48e54f3d03f657344fe60bd04f6898327ac6039b2efbbe4d2c015173f"),
)

def load_archived():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The archived benchmark requires CPython 3.12")
    for module_name, filename, expected in MODULES:
        path = HISTORICAL / (filename + ".cpython312")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived module failed integrity check: {path.name}")
        loader = importlib.machinery.SourcelessFileLoader(module_name, str(path))
        spec = importlib.util.spec_from_loader(module_name, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loader.exec_module(module)
    return module

def main():
    import os
    import torch

    torch.set_num_threads(int(os.environ.get("EINN_CPU_THREADS", "4")))
    archived = load_archived()
    args = archived.parse_args()
    if args.out_root.exists():
        raise SystemExit(f"Choose a new output directory: {args.out_root}")
    archived.main()

if __name__ == "__main__":
    main()
