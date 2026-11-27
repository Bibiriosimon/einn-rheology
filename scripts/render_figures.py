"""Render the seven data-driven submission figures in a separate output directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    target = args.destination.resolve()
    target.mkdir(parents=True, exist_ok=False)
    for number in (1, 2, 3, 4, 5, 6, 8):
        name = f"Figure_{number}"
        shutil.copytree(ROOT / "figures" / name, target / name)
        script = target / name / "scripts" / f"plot_figure_{number}.py"
        subprocess.run([sys.executable, str(script)], check=True)
        rendered = target / name / f"{name}.pdf"
        if not rendered.is_file() or rendered.stat().st_size == 0:
            raise ValueError(f"Rendering did not produce {rendered}")
        print(rendered.relative_to(target), flush=True)

if __name__ == "__main__":
    main()
